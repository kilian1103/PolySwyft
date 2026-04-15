# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2026-04-15

### Fixed

- platform-specific torch caps for macOS x86_64 compatibility (#11)

## [0.4.1] - 2026-04-15

### Fixed

- backfill CHANGELOG.md with accurate version history (#10)

## [0.4.0] - 2026-04-15

### Added

- automate changelog generation with git-cliff (#8)


### Fixed

- force-recreate tag in version-bump to handle rerun after failure (#9)

## [0.2.3] - 2026-04-15

### Fixed

- split branch and tag push in version-bump to trigger publish workflow (#7)

## [0.2.2] - 2026-04-15

### Fixed

- add PyPI version badge to README (#6)

## [0.2.0] - 2026-04-15

### Added

- add PyPI publish workflow and update package classifiers (#4)

## [0.1.2] - 2026-04-14

### Fixed

- add checkout step to claude-review workflow

## [0.1.1] - 2026-04-14

### Changed

- add CLAUDE.md and create_pr Claude Code skill (#6)

- add CLAUDE.md and create_pr Claude Code skill (#1)


### Fixed

- add github_token to claude-review workflow

- use env var instead of secrets context in version-bump step condition


