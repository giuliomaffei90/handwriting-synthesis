"""Dev tool: write hw/previews.npz, the sample line shown in the style picker.

Every style previews the same sentence, so the picker compares hands rather
than whatever each writer happened to be copying. Generating a line takes about
a second, which is too slow to do while someone flips through the styles, so it
is done once here and the result ships with the app.

Run: .venv/bin/python tools/make_previews.py
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hw.engine import Model, available_styles, style_strokes, style_text  # noqa: E402

SENTENCE = 'The quick brown fox jumps over the lazy dog'
# has to match the app's own default, or the picker advertises a neatness
# nobody gets - see DEFAULT_BIAS in Composer.swift
BIAS = 1.8
SEED = 20260910
TRIES = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', type=int, nargs='*',
                        help='redo just these styles, keeping the rest')
    parser.add_argument('--shift', type=int, default=0,
                        help='different attempts, for a style whose preview came out wrong')
    parser.add_argument('--attempt', type=int,
                        help='take exactly this attempt instead of scoring them - the escape '
                             'hatch for a hand whose letters come out wrong at a length the '
                             'score cannot see')
    args = parser.parse_args()

    out = os.path.join(ROOT, 'hw', 'previews.npz')
    previews = {}
    if args.only and os.path.exists(out):
        with np.load(out) as existing:
            previews = {k: existing[k] for k in existing.files if k.startswith('style')}

    model = Model()
    wanted = args.only if args.only else available_styles()
    for style in wanted:
        # a style can still lose the thread on a given attempt, and this line is
        # the one people judge the hand by, so write it a few times and keep the
        # one closest to the pace the hand should write at. Not the shortest:
        # rambling runs long, but a line that gives up half way runs short, and
        # picking the minimum chooses those.
        pace = len(style_strokes(style)) / float(len(style_text(style)))
        best, score, best_ratio = None, float('inf'), 0.0
        attempts = [args.attempt] if args.attempt is not None else range(TRIES)
        for attempt in attempts:
            strokes = model.generate([SENTENCE], style=style, bias=BIAS,
                                     seed=SEED + 1000 * (attempt + args.shift) + style)[0]
            ratio = len(strokes) / (len(SENTENCE) * pace)
            if abs(ratio - 1.0) < score:
                best, score, best_ratio = strokes, abs(ratio - 1.0), ratio
        previews['style{}'.format(style)] = best.astype(np.float32)
        print('  style {:>2}  {:>4} points  ratio {:.2f}'.format(style, len(best), best_ratio))

    out = os.path.join(ROOT, 'hw', 'previews.npz')
    np.savez_compressed(out, sentence=np.array(SENTENCE.encode('utf-8')), **previews)
    print('wrote {} ({:.0f} KB)'.format(out, os.path.getsize(out) / 1e3))


if __name__ == '__main__':
    main()
