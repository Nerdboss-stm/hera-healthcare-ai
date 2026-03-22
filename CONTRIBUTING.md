# Contributing to HERA Healthcare AI

Thanks for your interest in contributing to HERA! This document explains how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/hera-healthcare-ai.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Start PostgreSQL: `docker compose up postgres -d`
5. Run tests: `pytest tests/`
6. Start the API: `uvicorn serving.api:app --reload`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run linting: `ruff check .`
4. Run tests: `pytest tests/ -v`
5. Commit with a clear message: `git commit -m "feat: add patient export endpoint"`
6. Push and open a Pull Request

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `test:` — Adding or updating tests
- `refactor:` — Code change that neither fixes a bug nor adds a feature
- `ci:` — CI/CD changes

## Code Standards

- Python 3.10+
- Type hints on all public functions
- Docstrings on all modules and classes
- Files under 500 lines
- All clinical logic must have corresponding tests
- Never hardcode credentials or API keys

## Clinical Data Rules

- **NEVER** commit real patient data (PHI/PII)
- All test data must be synthetic
- Follow HIPAA Safe Harbor de-identification guidelines
- Clinical logic changes require review by a domain expert

## Pull Request Process

1. Update tests for any new functionality
2. Ensure CI passes (linting + tests)
3. Add a clear description of what and why
4. Request review from a maintainer
5. Squash and merge after approval

## Reporting Bugs

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant logs or error messages

## Questions?

Open a Discussion or Issue — we're happy to help.
