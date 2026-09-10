"""Stroke geometry and text preparation.

Trimmed version of the upstream drawing.py: same math, but without the
matplotlib/scipy dependencies (a 7-point Savitzky-Golay smoother is a fixed
convolution kernel, and the plotting helpers are unused by the app).
"""
import unicodedata
from collections import defaultdict

import numpy as np

alphabet = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]
alpha_to_num = defaultdict(int, list(map(reversed, enumerate(alphabet))))

MAX_STROKE_LEN = 1200
MAX_CHAR_LEN = 75

# characters the model never saw, mapped to the closest thing it did see
SUBSTITUTIONS = {
    '’': "'", '‘': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '…': '...', '·': '.',
    ' ': ' ', '\t': ' ',
    'Q': 'q', 'X': 'x', 'Z': 'z',  # missing uppercase in the training set
    '&': 'and', '/': '-', '*': '.', '_': '-', '€': 'EUR', '$': 'S',
    '%': '.', '=': '-', '+': 't', '@': 'a', 'ß': 'ss',
}


def encode_ascii(ascii_string):
    """encodes ascii string to array of ints"""
    return np.array([alpha_to_num[c] for c in ascii_string] + [0])


# grave and acute, the accents an Italian keyboard actually types
ACCENTS = {'\u0300', '\u0301'}


def sanitize(text):
    """Map arbitrary text onto the model's 73 character alphabet.

    An accented letter becomes the plain letter and an apostrophe, the way it
    is typed on a machine that has no accents: e' for e-grave. Other marks -
    cedillas, tildes, diaereses - are simply dropped, since an apostrophe would
    be wrong there.

    Returns (clean_text, dropped) where `dropped` is the sorted set of
    characters that had no usable equivalent and were removed.
    """
    valid = set(alphabet)
    out, dropped = [], set()
    for char in text:
        if char in valid or char == '\n':
            out.append(char)
            continue
        repl = SUBSTITUTIONS.get(char)
        if repl is None:
            decomposed = unicodedata.normalize('NFD', char)
            base = ''.join(c for c in decomposed if not unicodedata.combining(c))
            marks = {c for c in decomposed if unicodedata.combining(c)}
            if base and all(c in valid for c in base):
                repl = base + "'" if marks & ACCENTS else base
            else:
                # last resort for the likes of a ligature or a full-width digit
                folded = unicodedata.normalize('NFKD', char)
                repl = ''.join(c for c in folded if not unicodedata.combining(c))
                if not repl or not all(c in valid for c in repl):
                    repl = None
        if repl is None or not all(c in valid or c == '\n' for c in repl):
            dropped.add(char)
            continue
        out.append(repl)
    return ''.join(out), sorted(dropped)


def wrap(text, width=MAX_CHAR_LEN):
    """Split text into lines of at most `width` chars, breaking on spaces.

    Explicit newlines are kept (an empty line becomes vertical whitespace).
    """
    lines = []
    for paragraph in text.split('\n'):
        words, current = paragraph.split(' '), ''
        for word in words:
            while len(word) > width:  # a single word longer than a line
                if current:
                    lines.append(current)
                    current = ''
                lines.append(word[:width])
                word = word[width:]
            candidate = word if not current else current + ' ' + word
            if len(candidate) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def align(coords):
    """corrects for global slant/offset in handwriting strokes"""
    coords = np.copy(coords)
    X, Y = coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)
    X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
    offset, slope = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y).squeeze()
    theta = np.arctan(slope)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    coords[:, :2] = np.dot(coords[:, :2], rotation_matrix) - offset
    return coords


# Savitzky-Golay, window 7, polynomial order 3, mode='nearest'.
# For a 7-point cubic fit the smoothing coefficients are fixed, so the whole
# filter is one convolution against edge-padded data - no scipy needed.
_SAVGOL_7_3 = np.array([-2., 3., 6., 7., 6., 3., -2.]) / 21.0


def savgol(x):
    padded = np.pad(np.asarray(x, dtype=np.float64), 3, mode='edge')
    return np.convolve(padded, _SAVGOL_7_3, mode='valid')


def denoise(coords):
    """smoothing filter to mitigate some artifacts of the data collection"""
    strokes = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    new_coords = []
    for stroke in strokes:
        if len(stroke) == 0:
            continue
        xy = np.stack([savgol(stroke[:, 0]), savgol(stroke[:, 1])], axis=1)
        new_coords.append(np.concatenate([xy, stroke[:, 2].reshape(-1, 1)], axis=1))
    return np.vstack(new_coords)


def coords_to_offsets(coords):
    """convert from coordinates to offsets"""
    offsets = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    return np.concatenate([np.array([[0.0, 0.0, 1.0]]), offsets], axis=0)


def offsets_to_coords(offsets):
    """convert from offsets to coordinates"""
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)
