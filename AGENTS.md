# Repository Guidelines

## Project Structure & Module Organization
- `flash_attn/` core Python package; CUDA-backed ops live under `ops`, while `layers`, `models`, and `modules` provide higher-level building blocks.
- `csrc/` hosts CUDA/C++ kernels and wrappers grouped by backend (`flash_attn`, `flash_attn_ck`, `cutlass`, etc.); update headers and build scripts together.
- `hopper/` contains the FlashAttention-3 beta with its own `setup.py` and tests for Hopper GPUs.
- `tests/` mirrors the package layout with `pytest` suites plus top-level smoke files such as `test_flash_attn.py`.
- `benchmarks/`, `examples/`, and `training/` supply profiling scripts, usage demos, and training configs—treat them as references for API expectations.
- `assets/` stores documentation media; minimize binary diffs and compress new images.

## Build, Test, and Development Commands
- Inside `./venv/`, `pip install -e . --no-build-isolation` builds CUDA extensions against your pinned PyTorch/CUDA stack for iterative development.
- `MAX_JOBS=4 pip install -e . --no-build-isolation` caps parallel compilation on memory-bound machines.
- `python setup.py install` (run after `source venv/bin/activate`) produces a release-style build identical to the published wheel.
- `pytest -q` exercises the full regression suite; ensure a clean run before opening a PR.
- `cd hopper && PYTHONPATH=$PWD pytest -q -s test_flash_attn.py` validates FlashAttention-3 kernels on H100-class hardware.

## Coding Style & Naming Conventions
- Python follows PEP 8 with `black`/`ruff` configs (line length 100, Python ≥3.9); run `black flash_attn tests` and `ruff check flash_attn tests` before submitting.
- Use snake_case for functions/modules, CamelCase for classes, and keep CUDA binding names aligned with their `.cu` kernels for traceability.
- CUDA/C++ code in `csrc/` sticks to 2-space indents with namespace guards; preserve include order (`torch`, project headers, then system) and prefer descriptive template parameter names.

## Testing Guidelines
- Place new tests under `tests/<area>/test_<feature>.py`, mirroring the package structure for discoverability.
- Prefer parametrized `pytest` cases for shape and dtype sweeps; gate GPU-specific logic with capability checks (`pytest.importorskip("torch")` or `pytest.skip`).
- Validate numerical parity against PyTorch reference implementations (e.g., `flash_attn.ops.triton`) before benchmarking and document tolerances in assertions.

## Commit & Pull Request Guidelines
- Commits typically use bracketed scopes (for example, `[Cute,Bwd,Sm100] Fix grad launch`); keep tag sets tight and summaries in the imperative mood.
- Reference issue IDs in commit bodies when relevant, include performance or memory deltas for kernel changes, and note hardware/driver versions.
- Pull requests should capture motivation, affected modules, test hardware, and validation commands; attach profiles or screenshots when behaviour or performance changes.
