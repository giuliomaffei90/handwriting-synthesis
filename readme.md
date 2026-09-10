# Handwriting

A native macOS app that writes what you type in a human hand, and saves it as
PNG or SVG. No terminal, no Python, no TensorFlow - a 15 MB app that opens
instantly and works offline.

![](docs/sample.png)

This is a fork of [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis),
an implementation of the handwriting synthesis experiments in
[Generating Sequences with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
by Alex Graves. The model and its pretrained weights are unchanged. What is new
is everything around them: the network was lifted off TensorFlow onto numpy,
then onto Swift, so it fits in a double-clickable app.

## Download

1. Get `Handwriting-macOS-arm64.zip` from [Releases](../../releases).
2. Unzip and drag **Handwriting.app** into Applications.
3. First launch only: the app is signed but not notarized by Apple, so macOS
   asks. Right-click the app and choose **Open**, or go to
   **System Settings → Privacy & Security** and press **Open Anyway**.

Apple Silicon, macOS 13 or later.

## Using it

- **Type** anything. Lines longer than 75 characters wrap on word boundaries,
  blank lines become blank lines.
- **Style** picks one of 14 real handwriting samples that prime the network.
  The strip under the controls shows the actual sample.
- **Neatness** steadies the hand. It is the network's sampling bias: at 0.3 the
  writing wanders and scrawls, at 2.0 it writes carefully. The default, 1.0, is
  legible but still looks handwritten.
- **Pen** sets stroke width, and the well next to it sets the colour.
- **Save PNG** or **Save SVG**. Both are saved with a transparent background, so
  handwriting drops straight onto a document, and the SVG scales forever and
  opens in Illustrator or Figma.

The preview draws through the same code that writes the file, so what is on
screen is what gets saved. A line takes about a second.

The network only knows 73 characters. Accents are stripped (`è` becomes `e`),
a few symbols are substituted (`&` becomes `and`), uppercase `Q X Z` were never
in the training set and become lowercase. Anything left over is listed under the
text box instead of being silently dropped.

Occasionally a line comes out with invented words on the end, or loses the
thread halfway. The app catches the first case and writes the line again; the
second is the model itself and stays until someone retrains it. Style 9 is the
most reliable, style 2 the least. If a line looks wrong, press Write again.

![](docs/styles.png)

## Where the styles come from

A style is not a trained thing. There is one model, and a style is simply a
line of real handwriting shown to it before it starts writing - two files, the
pen movements and the text that was written. The model copies the hand it was
just shown.

The thirteen that ship with the original are nothing special: they are lines
0, 5, 8, 12, 19, 26, 28, 32, 37, 42, 44, 46 and 51 of the preprocessed IAM
On-Line Handwriting Database, the corpus the model was trained on - roughly
10,000 lines written by 221 people, each labelled with who wrote it. Anyone
with that corpus can pull out as many more as they like, and pick the ones
that behave. `prepare_data.py` needs only numpy to build it, no TensorFlow.

**Style 13** was added here from
[jonathanmaxberman's fork](https://github.com/jonathanmaxberman/handwriting-synthesis)
(MIT), the only one of the 609 forks of the original to contribute a new one.
It is a large, round, printed hand unlike anything in the original thirteen.
Its licence file carries a third party's copyright line, so treat its
provenance as best-effort: it is someone's handwriting sample, published under
MIT, reproduced here with credit.

To add your own, the recipe - independently arrived at by three people in the
upstream issues - is: record the pen positions, flip the y axis, then
`align`, `denoise`, `coords_to_offsets` and `normalize` from `drawing.py`, cut
to 1200 points, and save the offsets next to the text you wrote. Be warned
that priming steers the model rather than cloning your hand: it picks up size,
slant and roundness, not your letterforms.

## How it runs without TensorFlow

The original needs TensorFlow 1.6, which has no Apple Silicon build. So the
model was moved off it entirely, in two steps.

**Off TensorFlow.** `tools/tf_export.py` loads the frozen meta graph with TF2's
`compat.v1` API - no `tf.contrib`, no Python 2, no Rosetta - and dumps the ten
weight tensors the model actually uses into `hw/weights.npz`: 14 MB, down from a
43 MB checkpoint that was mostly optimizer state. `hw/engine.py` then
re-implements inference in numpy: three 400-unit LSTMs, the Gaussian attention
window over the character sequence, and the 20-component mixture density output
that gets sampled into pen movements. `hw/drawing.py` drops scipy, because the
Savitzky-Golay smoother the original imports is a fixed 7-tap kernel.

**Off Python.** `swift/` is the app: the same network again in Swift, matrix
products through Accelerate, drawn and exported with Core Graphics. numpy stays
as the reference implementation - it is what the Swift port is checked against,
and what the tools use.

Both ports are verified the same way, against a trace captured from the original
TensorFlow graph. After 738 recurrent steps the mixture density parameters agree
to `7.6e-05` in numpy and `1.75e-04` in Swift, on values up to 25, and every
state tensor to `1e-05`. `tests/test_engine.py` checks the numpy side;
`Handwriting --selftest` checks the app, and the build refuses to ship a bundle
that fails it.

The sampling path was checked separately by running numpy and TensorFlow at a
bias high enough to make sampling near-deterministic: same trajectory, identical
pen-up flags on all 203 steps.

### Knowing when to stop

The model decides it has finished a line when the pen happens to lift at the
exact step its attention reaches the last character. Styles that rarely lift the
pen never hit that coincidence: the attention gets stuck a few characters short
and the model rewrites the last word until the step budget runs out, which is
where invented words come from. Two changes catch it. The budget is no longer a
flat 40 timesteps per character but 1.45 times the pace of the chosen style,
measured from its own priming sample - styles write at between 19 and 42
timesteps per character, so the old fixed budget was generous for some and
impossible for others. And a line that does not finish inside that budget is
written again, up to five times, keeping the attempt that read furthest. Over a
hundred healthy lines none needed more than 1.39 times their style's pace, while
stuck ones ran to 2.2, 3.8 and beyond, so the two separate cleanly.

## Building it yourself

```bash
./tools/build_app.sh          # -> dist/Handwriting.app
```

It needs Xcode's Swift toolchain, and [uv](https://docs.astral.sh/uv/) for the
one Python step that packs `hw/weights.npz` and the styles into the binary the
app reads. `tools/build_dev_mac.sh` does the same and then drops the build in
~/Downloads and relaunches it.

The app is also a small command line tool, which is how it gets checked without
clicking through it:

```bash
dist/Handwriting.app/Contents/MacOS/Handwriting --selftest
dist/Handwriting.app/Contents/MacOS/Handwriting --write "hello there" --style 9 --out page
```

For the numpy side:

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install numpy pillow
.venv/bin/python tests/test_engine.py
```

`hw/weights.npz` is committed, so none of this needs TensorFlow. Regenerating it
from `checkpoints/` does - `uv pip install "tensorflow>=2.16"`, then
`python tools/tf_export.py`. `tools/make_icon.py` draws the app icon by asking
the model to write "Aa", and `tools/make_samples.py` builds the images above.

## The original project

The training pipeline is untouched and still needs TensorFlow 1.x:
`prepare_data.py` for the IAM-OnDB data, `rnn.py` to train, `demo.py` for the
scripted examples. See the
[upstream README](https://github.com/sjvasquez/handwriting-synthesis/blob/master/readme.md)
and its [web demo](https://seanvasquez.com/handwriting-generation/).
All credit for the model, the training and the pretrained weights goes to
[Sean Vasquez](https://github.com/sjvasquez).
