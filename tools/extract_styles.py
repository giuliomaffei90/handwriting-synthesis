"""Dev tool: mine new styles out of the IAM On-Line corpus.

A style is one line of real handwriting the model is primed with, so any line
of the corpus it was trained on is a candidate. Most are fine, some send the
model off the rails, and many look alike - so candidates are screened by
actually writing with them, and the survivors are chosen for variety rather
than taken in file order.

    python tools/extract_styles.py --strokes data/strokes-py3.npy \
        --sentences data/sentences.txt --pool 120 --keep 12

Nothing is written until --write is passed.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hw import drawing  # noqa: E402
from hw.engine import (Model, STEP_BUDGET, available_styles, style_strokes,  # noqa: E402
                       style_text)

# the sentences every candidate is asked to write while being screened - a hand
# that only manages one of them is not one to keep
PROBES = ['the quick brown fox jumps over it', 'Everything is going to be alright.']


def candidates(strokes_path, sentences_path):
    """Corpus lines in the model's own format, dropping the unusable ones."""
    raw = np.load(strokes_path, allow_pickle=True, encoding='latin1')
    sentences = open(sentences_path).read().split('\n')

    out = []
    for index, (sample, sentence) in enumerate(zip(raw, sentences)):
        text, dropped = drawing.sanitize(sentence.strip())
        if dropped or not (8 <= len(text) <= drawing.MAX_CHAR_LEN):
            continue
        # the corpus stores (pen_up, dx, dy); the model wants (dx, dy, pen_up)
        offsets = np.stack([sample[:, 1], sample[:, 2], sample[:, 0]], axis=1)
        offsets = offsets[:drawing.MAX_STROKE_LEN].astype(np.float32)
        if len(offsets) < 200:
            continue
        out.append((index, text, offsets))
    return out


def describe(text, offsets):
    """A few numbers that separate one hand from another."""
    coords = drawing.offsets_to_coords(offsets.astype(np.float64))
    height = np.percentile(coords[:, 1], 95) - np.percentile(coords[:, 1], 5)
    steps = offsets[:, :2]
    angles = np.arctan2(steps[:, 1], steps[:, 0])
    turn = np.abs(np.diff(np.unwrap(angles)))
    return np.array([
        height,                                   # how tall the hand is
        len(offsets) / float(len(text)),          # how long it dwells per character
        offsets[:, 2].mean(),                     # how often the pen leaves the paper
        np.median(turn),                          # how curly it is
        np.polyfit(coords[:, 0], coords[:, 1], 1)[0],   # slant
    ])


def writes_cleanly(model, text, offsets, seeds, bias):
    """Prime the model with a candidate and see whether it finishes the probes.

    Every probe at every seed has to come out, because a hand that works once
    and stalls the next time is worse than useless in a picker.
    """
    pace = len(offsets) / float(len(text))
    ratios = []
    for probe in PROBES:
        encoded = drawing.encode_ascii(text + ' ' + probe)
        c = encoded[None, :].astype(np.int32)
        c_len = np.array([len(encoded)], dtype=np.int32)
        context = model.context(c, c_len)
        primed = model.teacher_force(
            offsets[None], np.array([len(offsets)], np.int32), c, c_len)[1]
        limit = int(np.ceil(STEP_BUDGET * pace * len(probe)))

        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            bias_vector = np.full([1], bias, dtype=np.float32)
            first = model.sample_point(primed, bias_vector, rng)
            run = model.free_run(primed, context, bias_vector, limit, rng, first_input=first)
            if not run[1][0]:
                return None
            ratios.append(len(run[0][0]) / (len(probe) * pace))
    return float(np.mean(ratios))


def spread(descriptors, count, already_have):
    """Greedy farthest-point picking, so the chosen hands differ from each
    other and from the styles the app already ships."""
    everything = np.vstack([already_have, descriptors]) if len(already_have) else descriptors
    centre, scale = everything.mean(0), everything.std(0) + 1e-8
    pool = (descriptors - centre) / scale
    taken = [(already_have - centre) / scale] if len(already_have) else []

    chosen = []
    while len(chosen) < min(count, len(pool)):
        if taken:
            reference = np.vstack(taken)
            distances = np.min(
                np.linalg.norm(pool[:, None, :] - reference[None, :, :], axis=2), axis=1)
        else:
            distances = np.linalg.norm(pool - pool.mean(0), axis=1)
        distances[chosen] = -1
        pick = int(np.argmax(distances))
        chosen.append(pick)
        taken.append(pool[pick][None])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strokes', required=True)
    parser.add_argument('--sentences', required=True)
    parser.add_argument('--pool', type=int, default=120, help='candidates to screen')
    parser.add_argument('--keep', type=int, default=12, help='styles to choose')
    parser.add_argument('--seeds', type=int, default=4, help='probes per candidate')
    parser.add_argument('--max-ratio', type=float, default=1.15,
                        help='reject a hand that dwells longer than this on the probe')
    parser.add_argument('--bias', type=float, default=1.0)
    parser.add_argument('--offset', type=int, default=0, help='where to start in the corpus')
    parser.add_argument('--write', action='store_true', help='actually save the style files')
    args = parser.parse_args()

    model = Model()
    pool = candidates(args.strokes, args.sentences)[args.offset:args.offset + args.pool]
    print('screening {} candidates by writing with each one...'.format(len(pool)))

    survivors = []
    for position, (index, text, offsets) in enumerate(pool):
        ratio = writes_cleanly(model, text, offsets, args.seeds, args.bias)
        if ratio is not None and ratio < args.max_ratio:
            survivors.append((index, text, offsets, ratio))
        if (position + 1) % 20 == 0:
            print('  {}/{} screened, {} usable'.format(position + 1, len(pool), len(survivors)))
    print('{} of {} candidates write cleanly'.format(len(survivors), len(pool)))
    if not survivors:
        return

    descriptors = np.array([describe(t, o) for _, t, o, _ in survivors])
    existing = np.array([describe(style_text(s), style_strokes(s)) for s in available_styles()])
    picks = spread(descriptors, args.keep, existing)

    next_id = max(available_styles()) + 1
    print('\nchosen (corpus line -> style id):')
    for rank, position in enumerate(picks):
        index, text, offsets, ratio = survivors[position]
        style_id = next_id + rank
        print('  line {:>5} -> style {:<3} ratio {:.2f}  {:>4} points  "{}"'.format(
            index, style_id, ratio, len(offsets), text[:42]))
        if args.write:
            np.save(os.path.join(ROOT, 'styles', 'style-{}-strokes.npy'.format(style_id)),
                    offsets)
            np.save(os.path.join(ROOT, 'styles', 'style-{}-chars.npy'.format(style_id)),
                    np.array(text.encode('utf-8')))
    if not args.write:
        print('\n(nothing written - pass --write to save these)')


if __name__ == '__main__':
    main()
