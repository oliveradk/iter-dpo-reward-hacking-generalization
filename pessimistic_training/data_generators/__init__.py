"""Training data generators: frozen configs bundling both pipeline stages.

A generator bundles the `generate` + `select` stage configs under one
`TrainingDataGenerator`. There is NO in-process registry: a generator is
loaded from a JSON config file with `load_generator(path)`. Config files are
partial *overlays* over an optional `base` (see `base.py` for the format);
the per-env base configs live in `configs/`.
"""

from pessimistic_training.data_generators.base import (
    TrainingDataGenerator,
    load_generator,
)

__all__ = [
    "TrainingDataGenerator",
    "load_generator",
]
