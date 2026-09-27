# Credentials

This directory contains documentation and examples only. Store real credentials outside the
repository, normally under `~/.config/codeaction/`, with file mode `0600`.

The controller accepts an explicit credential file and mounts it read-only at
`/run/secrets/codeaction/provider.env`. Model identity, context limits, budgets, and rate-limit
policy remain committed non-secret configuration and cannot be overridden by the credential file.

See [authentication setup](../docs/authentication.md) for exact key names and vendor login steps.
