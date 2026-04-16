# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VoxKitchen, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email the maintainers directly. We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Scope

VoxKitchen processes audio files and downloads datasets from external sources. Security concerns include:

- **Path traversal** in audio file handling or tar extraction
- **Code injection** via malicious YAML pipeline files
- **Dependency vulnerabilities** in third-party packages

## Supported Versions

| Version | Supported |
|---------|:---------:|
| latest main branch | Yes |
| < 0.1.0 | No |
