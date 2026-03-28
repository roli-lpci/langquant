# Contributing to LangQuant

Thanks for your interest! LangQuant is a research project by [Hermes Labs](https://hermes-labs.ai).

## Getting Started

```bash
git clone https://github.com/roli-lpci/langquant.git
cd langquant
pip install pytest pyitlib scipy numpy
```

## Running Tests

```bash
pytest -v
```

Note: Some tests require [Ollama](https://ollama.ai) with `qwen3.5:9b` and `qwen3.5:4b` models pulled locally. Tests that require Ollama will be skipped if it's not available.

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Run `ruff check .` and fix any issues
4. Run `pytest -v` and ensure tests pass
5. Open a PR with a clear description

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting
- Keep functions focused and well-named
- Add docstrings for public functions

## Research Contributions

If you're extending the experiments (new models, new scaffold conditions, scale tests), please:
- Include raw JSONL results in `results/`
- Document methodology in your PR description
- Update LOG.md with findings

## Questions?

Open an issue or email via GitHub Issues on this repository.
