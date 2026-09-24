# Contributing to Certus

Thank you for your interest in contributing to Certus.

## Getting started

1. Clone the repository and follow the setup instructions in `README.md`.
2. Create a feature branch from `main`.
3. Make your changes in small, focused commits.

## Commit messages

Use lowercase conventional prefixes without a colon:

```
feat add new endpoint for batch grading
fix correct fovea offset on messidor-2 images
docs update api readme with deployment notes
refactor extract tile pooling into separate module
chore bump fastapi dependency
```

## Code style

- **Python**: follow PEP 8, use type hints where practical, and add docstrings to public functions.
- **TypeScript**: run `npm run lint` before pushing.
- **MATLAB**: match the naming conventions already in `matlab/`.

## What not to commit

- Datasets (`/Data/`), research papers (`iclr/`), or large checkpoints (>50 MB).
- Build artifacts, logs (`*.log`, `*.err`, `*.aux`), or virtual environments (`.venv/`).
- The `.gitignore` is authoritative — run `git status` before committing.

## Testing

- Backend: `python smoke_test.py` from `api/`.
- Frontend: `npm run lint` from `web/`.

## Pull requests

Open a PR against `main` with a clear description of what changed and why. Link any related
issues. Reviewers will check that the `.gitignore` rules are respected and that no withheld
data is included.
