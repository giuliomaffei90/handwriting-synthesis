"""Numpy inference for the handwriting synthesis model (Graves 2013).

Mirrors rnn_cell.LSTMAttentionCell / rnn.rnn from the original TensorFlow
implementation, one timestep at a time:

    lstm1(w_prev, x) -> attention window w -> lstm2(x, h1, w) -> lstm3(x, h2, w)
    h3 -> 20-component mixture density output -> sampled (dx, dy, pen_up)

Weights come from hw/weights.npz (see tools/tf_export.py). Verified against
the original graph by tests/test_engine.py.
"""
import json
import os
import sys
from collections import namedtuple

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

# how far the attention must move for the pen to count as still making progress
STALL_PROGRESS = 0.5

# How far the reading may fall back before the line is written off. When the
# model loses the thread it stops advancing through the text and returns to
# characters it has already written, and what comes out of the pen from there on
# is gibberish. Over lines judged by eye, the ones that came out right never
# fell back by more than 2.9 characters while the good ones went to 5, 7 and 12,
# so 3 catches three quarters of the failures and rejects none of the successes.
LOST_BACKTRACK = 3.0

# The corpus the model learnt from is made of short lines: half of them are 29
# characters or fewer, only one in fifty reaches 45, and not one reaches 63. Ask
# for a line longer than that and it is being asked for something it has never
# seen, which is where most of the gibberish comes from. So a line is written in
# pieces of this length, each straightened on its own and set down on the same
# baseline as the last - which reads as one continuous line.
PIECE_WIDTH = 38
PIECE_GAP = 11.0        # model units between pieces, about a space wide

# Each style writes at its own pace, which its priming sample measures: style 12
# spends 19 timesteps on a character, style 8 spends 42. Over 100 lines that
# came out right, none needed more than 1.39 times that pace, while lines where
# the model got stuck rewriting a word ran to 2.2, 3.8 and beyond - so a budget
# just past the healthy worst case separates them. Cutting a good line short
# only costs another attempt; letting a stuck one through costs gibberish.
STEP_BUDGET = 1.45
DEFAULT_SPEED = 30.0   # timesteps per character, when writing without a style

Run = namedtuple('Run', 'samples finished last_progress read_at')


def sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def softplus(x):
    return np.logaddexp(x, 0.0)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def style_names():
    """Human names for the styles, from styles/names.json."""
    path = os.path.join(STYLES_DIR, 'names.json')
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return {int(k): v for k, v in json.load(handle).items()}


def style_name(style):
    return style_names().get(style, 'Style {}'.format(style))


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
        """Feed the model's own samples back in until every line is done.

        `finished` says which lines stopped because the attention ran off the
        end of their text, rather than because the step budget ran out or
        because the reading fell back and the line was written off;
        `last_progress` is the step at which each line's reading position last
        moved forward - where to cut a line stuck rewriting the same word - and
        `read_at` is how far into the text it got.
        """
        batch = state['h1'].shape[0]
        if first_input is None:
            # rnn.sample: the pen starts lifted at the origin
            x = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (batch, 1))
        else:
            x = first_input
        finished = np.zeros(batch, dtype=bool)
        samples = np.zeros([batch, max_steps, 3], dtype=np.float32)
        read_at = np.full(batch, -np.inf)
        read_max = np.full(batch, -np.inf)
        lost = np.zeros(batch, dtype=bool)
        last_progress = np.zeros(batch, dtype=int)

        for t in range(max_steps):
            state = self.step(x, state, ctx)
            x = self.sample_point(state, bias, rng)
            samples[:, t] = np.where((finished | lost)[:, None], 0.0, x)

            # which character the pen is reading, as the attention-weighted
            # mean of the window centres
            alpha, kappa = state['alpha'], state['kappa']
            position = (alpha * kappa).sum(axis=1) / np.maximum(alpha.sum(axis=1), 1e-8)
            moved_on = position > read_at + STALL_PROGRESS
            read_at = np.where(moved_on, position, read_at)
            last_progress = np.where(moved_on & ~finished, t, last_progress)

            # gone back to text it has already written, with more still to go
            read_max = np.maximum(read_max, position)
            lost |= ((position < read_max - LOST_BACKTRACK)
                     & (position < ctx['c_len'] - 2) & ~finished)

            # done when attention has passed the last character (and the pen
            # was just lifted), matching LSTMAttentionCell.termination_condition
            char_idx = np.argmax(state['phi'], axis=1)
            past_end = char_idx >= ctx['c_len']
            at_end = (char_idx >= ctx['c_len'] - 1) & (x[:, 2] == 1.0)
            finished |= (past_end | at_end) & ~lost
            if (finished | lost).all():
                samples = samples[:, :t + 1]
                break
            if progress is not None and t % 25 == 0:
                progress((t + 1) / float(max_steps))

        return Run(samples, finished, last_progress, read_at)

    # ------------------------------------------------------------------- api

    def _write(self, texts, style, bias, rng, attempts, progress=None):
        """Write one batch of short texts, retrying the ones that come out wrong."""
        prefix = style_text(style) + ' ' if style is not None else ''
        encoded = [drawing.encode_ascii(prefix + t) for t in texts]
        char_len = max(len(e) for e in encoded)
        c = np.zeros([len(texts), char_len], dtype=np.int32)
        c_len = np.zeros([len(texts)], dtype=np.int32)
        for i, enc in enumerate(encoded):
            c[i, :len(enc)] = enc
            c_len[i] = len(enc)

        biases = np.full([len(texts)], bias, dtype=np.float32)
        ctx = self.context(c, c_len)

        if style is not None:
            x_prime = style_strokes(style).astype(np.float32)
            x_p = np.tile(x_prime[None], (len(texts), 1, 1))
            x_p_len = np.full([len(texts)], len(x_prime), dtype=np.int32)
            _, state = self.teacher_force(x_p, x_p_len, c, c_len)
            pace = len(x_prime) / float(len(style_text(style)))
        else:
            state = self.zero_state(len(texts), char_len)
            pace = DEFAULT_SPEED

        max_steps = int(np.ceil(STEP_BUDGET * pace * max(len(t) for t in texts)))
        written = [None] * len(texts)
        read_best = np.full(len(texts), -np.inf)
        pending = np.arange(len(texts))

        for _attempt in range(max(1, attempts)):
            rows = pending
            sub_state = {k: v[rows] for k, v in state.items()}
            sub_ctx = {'values': ctx['values'][rows], 'u': ctx['u'], 'c_len': ctx['c_len'][rows]}
            sub_bias = biases[rows]
            # rnn.primed_sample continues from the primed state, so the first
            # input is sampled rather than the zero/pen-up vector
            first_input = self.sample_point(sub_state, sub_bias, rng) if style is not None else None

            run = self.free_run(sub_state, sub_ctx, sub_bias, max_steps, rng,
                                first_input=first_input, progress=progress)

            for i, row in enumerate(rows):
                if run.finished[i]:
                    sample = run.samples[i]
                elif run.read_at[i] > read_best[row]:
                    read_best[row] = run.read_at[i]
                    sample = run.samples[i][:run.last_progress[i] + 1]
                else:
                    continue          # an earlier attempt got further
                written[row] = sample[~np.all(sample == 0.0, axis=1)]

            pending = rows[~run.finished]
            if not len(pending):
                break
        return written

    def _join(self, pieces, gap=PIECE_GAP):
        """Set the pieces of one line down side by side on a shared baseline."""
        pieces = [p for p in pieces if len(p)]
        if not pieces:
            return np.zeros([0, 3], dtype=np.float32)
        if len(pieces) == 1:
            return pieces[0]

        placed, x = [], 0.0
        for piece in pieces:
            coords = drawing.offsets_to_coords(np.array(piece, dtype=np.float64))
            # straightening each piece on its own is what hides the joins: their
            # baselines end up on the same line instead of wandering apart
            coords[:, :2] = drawing.align(coords[:, :2])
            coords[:, 0] += x - coords[:, 0].min()
            coords[-1, 2] = 1.0        # lift the pen before the next piece
            placed.append(coords)
            x = coords[:, 0].max() + gap

        return drawing.coords_to_offsets(np.concatenate(placed)).astype(np.float32)

    def generate(self, lines, style=None, bias=0.75, seed=None, progress=None, attempts=5):
        """Generate handwriting for a list of text lines.

        Each line is written in pieces no longer than the lines the model was
        trained on and joined back together, and any piece that comes out wrong
        is written again - see PIECE_WIDTH and the retry rules above.

        Returns a list of [num_points, 3] offset arrays (dx, dy, pen_up), one
        per line; empty lines give an empty array.
        """
        rng = np.random.default_rng(seed)
        texts = list(lines)
        active = [i for i, line in enumerate(texts) if line.strip()]
        if not active:
            return [np.zeros([0, 3], dtype=np.float32) for _ in texts]

        # every piece of every line goes through the model together
        pieces, owner = [], []
        for index in active:
            for piece in drawing.wrap(texts[index], PIECE_WIDTH):
                if piece.strip():
                    pieces.append(piece)
                    owner.append(index)

        written = self._write(pieces, style, bias, rng, attempts, progress=progress)

        out = [np.zeros([0, 3], dtype=np.float32) for _ in texts]
        for index in active:
            out[index] = self._join([w for w, o in zip(written, owner) if o == index])
        return out
