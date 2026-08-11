# Security Policy

This document covers how to report a security vulnerability in this
repository. For an overview of Haki's security model (API key scoping,
Row-Level Security, the Policy Engine, what V1 does and does not cover),
see [docs/SECURITY.md](docs/SECURITY.md).

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

Use GitHub's private vulnerability reporting: go to the
[Security tab](https://github.com/GetHaki/Haki/security) of this
repository and select **"Report a vulnerability"**. This opens a private
disclosure channel with the maintainers only.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof of concept if applicable.
- The affected version or commit.

## Response

We aim to acknowledge reports within a few business days and to keep you
updated as we investigate and address the issue. Please give us reasonable
time to fix a confirmed vulnerability before any public disclosure.

## Scope

This applies to the code in this repository (the self-hostable core: API,
SDKs, MCP and n8n integrations). The hosted Cloud service has a separate,
private codebase — if you find an issue affecting the hosted service
specifically (not something you can reproduce by self-hosting), please
still report it here and we'll route it appropriately.
