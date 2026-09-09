"""Standalone numpy re-implementation of sjvasquez/handwriting-synthesis.

No TensorFlow: `tools/tf_export.py` converts the original checkpoint into
`hw/weights.npz` once, and `hw.engine` runs the model on numpy alone.
"""
