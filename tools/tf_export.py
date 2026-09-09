"""Dev tool (not shipped in the app): export the TF1 checkpoint to plain numpy.

Imports the frozen meta graph with TF2's compat.v1 API - no tf.contrib, no
Python 2, no Rosetta - then dumps:

  hw/weights.npz        the 6 weight tensors the model actually needs
  tests/reference.npz   a deterministic teacher-forced trace used by
                        tests/test_engine.py to prove the numpy port matches

Run with a venv that has tensorflow>=2.16:  python tools/tf_export.py
"""
import os
import sys

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf  # noqa: E402

tf1 = tf.compat.v1
tf1.disable_eager_execution()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, 'checkpoints', 'model-17900')

# variable name in checkpoint -> name in weights.npz
WEIGHTS = {
    'rnn/LSTMAttentionCell/lstm_cell/kernel': 'lstm1_kernel',
    'rnn/LSTMAttentionCell/lstm_cell/bias': 'lstm1_bias',
    'rnn/LSTMAttentionCell/lstm_cell_1/kernel': 'lstm2_kernel',
    'rnn/LSTMAttentionCell/lstm_cell_1/bias': 'lstm2_bias',
    'rnn/LSTMAttentionCell/lstm_cell_2/kernel': 'lstm3_kernel',
    'rnn/LSTMAttentionCell/lstm_cell_2/bias': 'lstm3_bias',
    'rnn/LSTMAttentionCell/attention/weights': 'attention_kernel',
    'rnn/LSTMAttentionCell/attention/biases': 'attention_bias',
    'rnn/gmm/weights': 'gmm_kernel',
    'rnn/gmm/biases': 'gmm_bias',
}

# placeholders, in graph creation order (see rnn.calculate_loss)
PH = dict(x='Placeholder:0', y='Placeholder_1:0', x_len='Placeholder_2:0',
          c='Placeholder_3:0', c_len='Placeholder_4:0',
          sample_tsteps='Placeholder_5:0', num_samples='Placeholder_6:0',
          prime='Placeholder_7:0', x_prime='Placeholder_8:0',
          x_prime_len='Placeholder_9:0', bias='PlaceholderWithDefault:0')

STATE_FIELDS = ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']


def main():
    sys.path.insert(0, ROOT)
    from hw import drawing

    reader = tf1.train.load_checkpoint(CKPT)
    weights = {name: reader.get_tensor(var) for var, name in WEIGHTS.items()}
    weights['alphabet_size'] = np.array(len(drawing.alphabet))
    os.makedirs(os.path.join(ROOT, 'hw'), exist_ok=True)
    out = os.path.join(ROOT, 'hw', 'weights.npz')
    np.savez(out, **weights)
    print('wrote {} ({:.1f} MB)'.format(out, os.path.getsize(out) / 1e6))

    # deterministic reference: teacher-force two padded style samples of
    # different lengths (exercises the sequence_length masking too)
    xs = [np.load(os.path.join(ROOT, 'styles', 'style-{}-strokes.npy'.format(i))) for i in (3, 9)]
    lens = np.array([len(xs[0]), len(xs[1]) - 37], dtype=np.int32)
    tmax = int(lens.max())
    x = np.zeros([2, tmax, 3], dtype=np.float32)
    for i, xi in enumerate(xs):
        x[i, :lens[i]] = xi[:lens[i]]

    texts = ['Hello world', 'the quick brown fox']
    cmax = max(len(t) for t in texts) + 1
    c = np.zeros([2, cmax], dtype=np.int32)
    c_len = np.zeros([2], dtype=np.int32)
    for i, t in enumerate(texts):
        enc = drawing.encode_ascii(t)
        c[i, :len(enc)] = enc
        c_len[i] = len(enc)

    g = tf1.Graph()
    with g.as_default():
        saver = tf1.train.import_meta_graph(CKPT + '.meta')
        params = g.get_tensor_by_name('rnn/gmm/add:0')
        state = [g.get_tensor_by_name('rnn/while/Exit_{}:0'.format(i + 3)) for i in range(11)]
        with tf1.Session() as sess:
            saver.restore(sess, CKPT)
            feed = {PH['x']: x, PH['x_len']: lens, PH['c']: c, PH['c_len']: c_len,
                    PH['bias']: np.zeros([2], dtype=np.float32)}
            ref_params, ref_state = sess.run([params, state], feed_dict=feed)

    os.makedirs(os.path.join(ROOT, 'tests'), exist_ok=True)
    ref = dict(x=x, x_len=lens, c=c, c_len=c_len, params=ref_params)
    ref.update({'state_' + k: v for k, v in zip(STATE_FIELDS, ref_state)})
    out = os.path.join(ROOT, 'tests', 'reference.npz')
    np.savez_compressed(out, **ref)
    print('wrote {} params={} ({:.1f} MB)'.format(out, ref_params.shape, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
