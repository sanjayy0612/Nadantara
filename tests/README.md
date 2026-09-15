# Tests

Tests mirror externally meaningful contracts rather than implementation file names.

- Audio tests validate format and byte transport.
- Model tests validate tensor shapes, masking, state reset and serialization.
- Training smoke tests use tiny configurations and synthetic tensors only.
- Performance numbers do not belong in unit tests; reproducible benchmarks use fixed
  manifests and are stored as artifacts outside Git.
