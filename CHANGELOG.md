# Changelog

All notable changes to this project are noted here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once it reaches 0.1.0.

## Unreleased

### Added
- A trace reader (`microagent.trace`) that prints a run as a plain story.
- A from-scratch walkthrough notebook that rebuilds all five blocks with a
  small-office analogy.
- Production notes (`notes/`) covering memory compaction, healing and retries,
  evaluation, prompt techniques, and going async.
- A reading-order guide and a contributing guide.
- Continuous integration on Python 3.10 to 3.13, plus ruff linting and
  formatting and a doctest check.
- An architecture diagram in the README.

## 0.0.1

### Added
- The core: `Message`, `Context`, `Tool`, `Policy`, and `Agent`, with zero
  runtime dependencies.
- An optional Ollama policy adapter built on the standard library only.
- Offline examples and a test suite.
