"""Proves the numpy port matches the original TensorFlow graph.

tests/reference.npz was produced by tools/tf_export.py running the restored
TF1 graph on fixed inputs; here the same inputs go through hw.engine and every
intermediate must agree. Run: python tests/test_engine.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hw import drawing  # noqa: E402
from hw.engine import (STEP_BUDGET, Model, STATE_FIELDS, style_strokes,  # noqa: E402
                       style_text)

REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference.npz')


def test_matches_tensorflow():
    ref = np.load(REFERENCE)
    model = Model()
    params, state = model.teacher_force(ref['x'], ref['x_len'], ref['c'], ref['c_len'])

    err = np.abs(params - ref['params']).max()
    scale = np.abs(ref['params']).max()
    print('mdn params: shape {} max abs err {:.2e} (values up to {:.1f})'.format(
        params.shape, err, scale))
    assert err < 1e-3, 'mixture density outputs diverge from TensorFlow: {}'.format(err)

    for field in STATE_FIELDS:
        expected = ref['state_' + field]
        err = np.abs(state[field] - expected).max()
        print('  state.{:<6s} max abs err {:.2e}'.format(field, err))
        assert err < 1e-3, 'state.{} diverges from TensorFlow: {}'.format(field, err)


def test_savgol_matches_scipy():
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print('scipy not installed, skipping savgol cross-check')
        return
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 7, 40):
        x = rng.standard_normal(n)
        err = np.abs(drawing.savgol(x) - savgol_filter(x, 7, 3, mode='nearest')).max()
        assert err < 1e-9, 'savgol mismatch at n={}: {}'.format(n, err)
    print('savgol matches scipy for n in (1, 2, 5, 7, 40)')


def test_lines_do_not_ramble():
    """A line must stop when its text is written, not carry on inventing words.

    Some styles rarely lift the pen, so the model's own stop condition - which
    needs a pen-up exactly as the attention reaches the last character - never
    fires, and it loops over the last word until the step budget runs out.
    Style 11 with this text did exactly that before the retry rule.
    """
    model = Model()
    for style, text, seed in [(11, 'Everything is going to be alright.', 3),
                              (9, 'the quick brown fox jumps over it', 3)]:
        pace = len(style_strokes(style)) / float(len(style_text(style)))
        budget = STEP_BUDGET * pace * len(text)
        points = len(model.generate([text], style=style, bias=1.0, seed=seed)[0])
        print('style {:>2}: {} points, budget {:.0f}'.format(style, points, budget))
        assert points <= budget, 'style {} rambled past its budget'.format(style)


def test_text_prep():
    # a non-breaking space has an equivalent and is substituted; a paragraph
    # separator does not, and is reported rather than vanishing silently
    text, dropped = drawing.sanitize(
        'Perch\u00e9 Qui: 3 \u201ctest\u201d \u2013 ok\u00a0\u2029')
    assert text == 'Perche qui: 3 "test" - ok ', repr(text)
    assert dropped == ['\u2029'], dropped
    assert all(len(line) <= 75 for line in drawing.wrap('word ' * 100))
    assert drawing.wrap('a\n\nb') == ['a', '', 'b']
    print('text preparation ok')


if __name__ == '__main__':
    test_matches_tensorflow()
    test_savgol_matches_scipy()
    test_lines_do_not_ramble()
    test_text_prep()
    print('OK')
