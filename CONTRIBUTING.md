# Contributing to trnfft

Thanks for your interest in contributing. This document covers the whole suite — the same guidelines apply to the umbrella and to each of the six sub-projects (`trnfft`, `trnblas`, `trnrand`, `trnsolver`, `trnsparse`, `trntensor`).

## Where does my change belong?

- **Scoped to a single library** — bug fix, new feature, added kernel in one sub-project → open a PR in that sub-project's repo.
- **Cross-cutting** — integration examples, umbrella docs, CUDA-rosetta content, coordinated version bumps, shared tooling → open a PR against [trnsci/trnsci](https://github.com/trnsci/trnsci).

If you aren't sure, open the PR where it's easiest and we'll move it.

## Development setup

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
# Clone sibling sub-projects if you don't already have them:
for p in trnfft trnblas trnrand trnsolver trnsparse trntensor; do
  [ -d "$p" ] || git clone "git@github.com:trnsci/$p.git"
done
make install-dev
make test-all
```

A single sub-project in isolation:

```bash
cd trnfft   # or any other
pip install -e ".[dev]"
pytest tests/ -v -m "not neuron"
```

## Tests

- `pytest tests/ -v -m "not neuron"` — the CPU-only suite. Must pass on every PR.
- `pytest tests/ -v -m neuron` — on-hardware suite, run from a machine with AWS credentials via `scripts/run_neuron_tests.sh`. Not part of standard CI.

## Conventions

- **License header** — new files get the Apache 2.0 notice and `Copyright 2026 Scott Friedman` where a notice is customary (long source files, not one-line configs).
- **Python** — ≥ 3.10; `torch` ≥ 2.1; `numpy` ≥ 1.24.
- **No emoji** in source, commits, or docs unless the feature is literally about emoji.
- **Docstrings** only where they help a reader who doesn't have the context you had when writing. Don't restate the signature.
- **No throwaway files** in the repo root — work-in-progress notes belong in PR descriptions or `docs/`.

## Commit style

Conventional-ish prefixes: `fix:`, `feat:`, `docs:`, `test:`, `bench:`, `chore:`, `refactor:`. Keep the subject line ≤ 72 characters. Longer context goes in the body.

## Pull requests

Before opening a PR:

- [ ] `pytest tests/ -v -m "not neuron"` passes
- [ ] If you changed public API or behaviour, `CHANGELOG.md` has an `[Unreleased]` entry
- [ ] If you added a public symbol, the project's `docs/api.md` lists it
- [ ] PR description says *what* changed and *why*, not just *what*

## Releasing

`.github/workflows/publish.yml` publishes to PyPI (trusted publishing) when a GitHub Release is published. A tag containing `-` (`vX.Y.Z-rc1`) goes to TestPyPI.

- [ ] Release PR (`release: vX.Y.Z`): bump `version` in `pyproject.toml` and `__version__` in `trnfft/__init__.py`, and move the `[Unreleased]` CHANGELOG entries under `## [X.Y.Z] - YYYY-MM-DD`.
- [ ] After the merge, run `gh release create vX.Y.Z --target <merge sha>`, then confirm the publish run and the new version on PyPI.
- [ ] **Update umbrella docs/extras** in one [trnsci/trnsci](https://github.com/trnsci/trnsci) PR:
  - if users should require this release (always for a fix to silent wrong results), bump `trnfft` in the umbrella `pyproject.toml` `[fft]` and `[all]` extras, and cut a `trnsci` meta-package release;
  - update the `[[described:trnfft=X.Y.Z]]` rows in `docs/roadmap.md` and `docs/nki_validation_status.md`, with the [#35](https://github.com/trnsci/trnsci/issues/35) Trn2 validation link;
  - check that `python scripts/check_docs_drift.py` exits 0. The weekly docs-drift workflow opens a `docs-stale` issue for anything missed.

## Issues

Label each issue one of: `bug`, `feature`, `question`, `docs`, `infra`. Bugs include repro steps and expected-vs-actual output. Features describe the use case.

## Code of conduct

By participating, you agree to abide by the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). Report concerns to the contact listed there.

## License

Contributions are Apache 2.0, per the repository's [LICENSE](LICENSE). You attest you have the right to submit the code.
