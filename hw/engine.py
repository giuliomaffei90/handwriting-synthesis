"""Numpy inference for the handwriting synthesis model (Graves 2013).

Mirrors rnn_cell.LSTMAttentionCell / rnn.rnn from the original TensorFlow
implementation, one timestep at a time:

    lstm1(w_prev, x) -> attention window w -> lstm2(x, h1, w) -> lstm3(x, h2, w)
    h3 -> 20-component mixture density output -> sampled (dx, dy, pen_up)

Weights come from hw/weights.npz (see tools/tf_export.py). Verified against
the original graph by tests/test_engine.py.
"""
import os
import sys

import numpy as np

from . import drawing


def resource_root():
    """Where data files live: the repo root, or the bundle when frozen."""
    bundled = getattr(sys, '_MEIPASS', None)
    if bundled:
        return bundled
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


WEIGHTS_PATH = os.path.join(resource_root(), 'hw', 'weights.npz')
STYLES_DIR = os.path.join(resource_root(), 'styles')

STATE_FIELDS = ('h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi')


def sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def softplus(x):
    return np.logaddexp(x, 0.0)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def style_text(style):
    """The text that was written to produce styles/style-N-strokes.npy."""
    path = os.path.join(STYLES_DIR, 'style-{}-chars.npy'.format(style))
    return np.load(path).tobytes().decode('utf-8')


def style_strokes(style):
    return np.load(os.path.join(STYLES_DIR, 'style-{}-strokes.npy'.format(style)))


def available_styles():
    return sorted(
        int(f.split('-')[1]) for f in os.listdir(STYLES_DIR) if f.endswith('-strokes.npy')
    )


class Model(object):

    def __init__(self, weights_path=WEIGHTS_PATH):
        with np.load(weights_path) as npz:
            self.w = {k: npz[k].astype(np.float32) for k in npz.files if k != 'alphabet_size'}
            alphabet_size = int(npz['alphabet_size'])
        if alphabet_size != len(drawing.alphabet):
            raise ValueError('weights were trained on a {}-character alphabet'.format(alphabet_size))
        self.alphabet_size = alphabet_size
        self.lstm_size = self.w['lstm1_bias'].shape[0] // 4
        self.num_attn = self.w['attention_bias'].shape[0] // 3
        self.num_mix = (self.w['gmm_bias'].shape[0] - 1) // 6

    # ------------------------------------------------------------------ cell

    def _lstm(self, n, x, h, c):
        z = np.concatenate([x, h], axis=1) @ self.w['lstm%d_kernel' % n] + self.w['lstm%d_bias' % n]
        i, j, f, o = np.split(z, 4, axis=1)
        c = sigmoid(f + 1.0) * c + sigmoid(i) * np.tanh(j)
        return sigmoid(o) * np.tanh(c), c

    def zero_state(self, batch_size, char_len):
        z = lambda n: np.zeros([batch_size, n], dtype=np.float32)  # noqa: E731
        state = {k: z(self.lstm_size) for k in ('h1', 'c1', 'h2', 'c2', 'h3', 'c3')}
        state.update({k: z(self.num_attn) for k in ('alpha', 'beta', 'kappa')})
        state['w'] = z(self.alphabet_size)
        state['phi'] = z(char_len)
        return state

    def context(self, c, c_len):
        """One-hot character values, pre-masked, plus the position index."""
        c, c_len = np.asarray(c), np.asarray(c_len)
        char_len = c.shape[1]
        values = np.eye(self.alphabet_size, dtype=np.float32)[c]
        mask = (np.arange(char_len) < c_len[:, None]).astype(np.float32)
        return {
            'values': values * mask[:, :, None],
            'u': np.arange(char_len, dtype=np.float32).reshape(1, 1, char_len),
            'c_len': c_len.astype(np.int32),
        }

    def step(self, x, state, ctx):
        w_prev = state['w']
        h1, c1 = self._lstm(1, np.concatenate([w_prev, x], axis=1), state['h1'], state['c1'])

        attn_in = np.concatenate([w_prev, x, h1], axis=1)
        alpha, beta, dkappa = np.split(
            softplus(attn_in @ self.w['attention_kernel'] + self.w['attention_bias']), 3, axis=1)
        kappa = state['kappa'] + dkappa / 25.0
        beta = np.maximum(beta, 0.01)

        phi = np.sum(
            alpha[:, :, None] * np.exp(-np.square(kappa[:, :, None] - ctx['u']) / beta[:, :, None]),
            axis=1,
        )
        w = np.einsum('bu,bua->ba', phi, ctx['values'])

        h2, c2 = self._lstm(2, np.concatenate([x, h1, w], axis=1), state['h2'], state['c2'])
        h3, c3 = self._lstm(3, np.concatenate([x, h2, w], axis=1), state['h3'], state['c3'])

        return dict(h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3,
                    alpha=alpha, beta=beta, kappa=kappa, w=w, phi=phi)

    # ---------------------------------------------------------------- output

    def mdn_params(self, h3):
        return h3 @ self.w['gmm_kernel'] + self.w['gmm_bias']

    def parse_params(self, params, bias, eps=1e-8, sigma_eps=1e-4):
        n = self.num_mix
        pis, sigmas, rhos, mus, es = np.split(params, [n, 3 * n, 4 * n, 6 * n], axis=-1)
        bias = np.asarray(bias, dtype=np.float32)[:, None]

        pis = softmax(pis * (1.0 + bias))
        pis = np.where(pis < 0.01, 0.0, pis)
        sigmas = np.clip(np.exp(sigmas - bias), sigma_eps, np.inf)
        rhos = np.clip(np.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = np.clip(sigmoid(es), eps, 1.0 - eps)
        es = np.where(es < 0.01, 0.0, es)
        return pis, mus, sigmas, rhos, es

    def sample_point(self, state, bias, rng):
        """Sample (dx, dy, pen_up) from the mixture density output."""
        pis, mus, sigmas, rhos, es = self.parse_params(self.mdn_params(state['h3']), bias)
        n, batch = self.num_mix, pis.shape[0]
        rows = np.arange(batch)

        cdf = np.cumsum(pis / pis.sum(axis=-1, keepdims=True), axis=-1)
        idx = np.minimum((cdf < rng.random((batch, 1))).sum(axis=-1), n - 1)

        mu1, mu2 = mus[rows, idx], mus[rows, idx + n]
        s1, s2, rho = sigmas[rows, idx], sigmas[rows, idx + n], rhos[rows, idx]
        z1, z2 = rng.standard_normal((2, batch))
        # cholesky of [[s1^2, rho s1 s2], [rho s1 s2, s2^2]]
        x1 = mu1 + s1 * z1
        x2 = mu2 + s2 * (rho * z1 + np.sqrt(1.0 - rho * rho) * z2)
        eos = (rng.random(batch) < es[:, 0]).astype(np.float32)
        return np.stack([x1, x2, eos], axis=1).astype(np.float32)

    # ----------------------------------------------------------------- runs

    def teacher_force(self, x, x_len, c, c_len, bias=None):
        """Run the model over given strokes (the training/priming path).

        Returns (mixture density parameters per timestep, final state), padded
        exactly like tf.nn.dynamic_rnn with sequence_length: frozen state and
        zero output past each sequence's length.
        """
        x = np.asarray(x, dtype=np.float32)
        x_len = np.asarray(x_len).astype(np.int32)
        batch, num_steps = x.shape[0], x.shape[1]
        ctx = self.context(c, c_len)
        state = self.zero_state(batch, ctx['u'].shape[2])

        outputs = np.zeros([batch, num_steps, self.lstm_size], dtype=np.float32)
        for t in range(num_steps):
            live = (t < x_len)[:, None]
            if not live.any():
                break
            new = self.step(x[:, t], state, ctx)
            state = {k: np.where(live, new[k], state[k]) for k in state}
            outputs[:, t] = np.where(live, state['h3'], 0.0)

        return self.mdn_params(outputs), state

    def free_run(self, state, ctx, bias, max_steps, rng, first_input=None, progress=None):
        """Feed the model's own samples back in until every line is done."""
        batch = state['h1'].shape[0]
        if first_input is None:
            # rnn.sample: the pen starts lifted at the origin
            x = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (batch, 1))
        else:
            x = first_input
        finished = np.zeros(batch, dtype=bool)
        samples = np.zeros([batch, max_steps, 3], dtype=np.float32)

        for t in range(max_steps):
            state = self.step(x, state, ctx)
            x = self.sample_point(state, bias, rng)
            samples[:, t] = np.where(finished[:, None], 0.0, x)

            # done when attention has passed the last character (and the pen
            # was just lifted), matching LSTMAttentionCell.termination_condition
            char_idx = np.argmax(state['phi'], axis=1)
            past_end = char_idx >= ctx['c_len']
            at_end = (char_idx >= ctx['c_len'] - 1) & (x[:, 2] == 1.0)
            finished |= past_end | at_end
            if finished.all():
                samples = samples[:, :t + 1]
                break
            if progress is not None and t % 25 == 0:
                progress((t + 1) / float(max_steps))

        return samples

    # ------------------------------------------------------------------- api

    def generate(self, lines, style=None, bias=0.75, seed=None, progress=None):
        """Generate handwriting for a list of text lines.

        Returns a list of [num_points, 3] offset arrays (dx, dy, pen_up), one
        per line; empty lines give an empty array.
        """
        rng = np.random.default_rng(seed)
        texts = [line for line in lines]
        active = [i for i, line in enumerate(texts) if line.strip()]
        if not active:
            return [np.zeros([0, 3], dtype=np.float32) for _ in texts]

        prefix = style_text(style) + ' ' if style is not None else ''
        encoded = [drawing.encode_ascii(prefix + texts[i]) for i in active]
        char_len = max(len(e) for e in encoded)
        c = np.zeros([len(active), char_len], dtype=np.int32)
        c_len = np.zeros([len(active)], dtype=np.int32)
        for i, enc in enumerate(encoded):
            c[i, :len(enc)] = enc
            c_len[i] = len(enc)

        biases = np.full([len(active)], bias, dtype=np.float32)
        ctx = self.context(c, c_len)

        if style is not None:
            x_prime = style_strokes(style).astype(np.float32)
            x_p = np.tile(x_prime[None], (len(active), 1, 1))
            x_p_len = np.full([len(active)], len(x_prime), dtype=np.int32)
            _, state = self.teacher_force(x_p, x_p_len, c, c_len)
            # rnn.primed_sample continues from the primed state, so the first
            # input is sampled rather than the zero/pen-up vector
            first_input = self.sample_point(state, biases, rng)
        else:
            state = self.zero_state(len(active), char_len)
            first_input = None

        max_steps = 40 * max(len(texts[i]) for i in active)
        samples = self.free_run(state, ctx, biases, max_steps, rng,
                                first_input=first_input, progress=progress)

        out = [np.zeros([0, 3], dtype=np.float32) for _ in texts]
        for i, line_idx in enumerate(active):
            sample = samples[i]
            out[line_idx] = sample[~np.all(sample == 0.0, axis=1)]
        return out
