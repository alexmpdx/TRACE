# Bug attachment storage

This orphan branch holds image attachments uploaded by TRACE's in-app
"Report a bug" feature. PNGs land in `bug-attachments/<uuid>.png` and
are linked from the GitHub issue that triggered the upload.

This branch is intentionally separate from `main` — it has no shared
history, so adding attachments here does not pollute the codebase or
its commit log. Developers cloning TRACE will not download these
attachments unless they explicitly check out this branch.

The Cloudflare Worker that creates the issues
(https://trace-bug-reporter.alexmpdx.workers.dev) writes here via the
GitHub Contents API using a server-side PAT.