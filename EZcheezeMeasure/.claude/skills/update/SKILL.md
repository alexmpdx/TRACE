---
name: update
description: End-of-session routine — push to GitHub, save a conversation log, and write a structured changes summary.
user_invocable: true
---

# /update — End-of-Session Update

When the user invokes `/update`, perform the following three steps in order.

## Step 1: Commit & Push to GitHub

1. Run `git status` to check for uncommitted changes (unstaged or staged).
2. **If there are uncommitted changes**, automatically commit them:
   - Stage all changes with `git add -A`.
   - Run `git diff --cached --stat` to see what will be committed.
   - Generate a concise commit message summarizing the changes made during this session.
   - Commit with that message (include `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`).
3. Run `git push origin <current-branch>` (detect the current branch with `git branch --show-current`).
4. Report the commit and push result to the user.

## Step 2: Export Conversation Log → `../Chat_files/`

Write a Markdown file to `../Chat_files/` (relative to the working directory, i.e. `/Users/alexmurphy/Desktop/claude_scripts/mapThemVeins/Chat_files/`).

- **Filename**: `YYYY-MM-DD-HHMMSS-<folder>-session-log.md` using the current date/time, where `<folder>` is the name of the current working directory (e.g. `WingVeinAnalyzer`). Detect with `basename "$PWD"`.
- **Content** — write from your memory of this conversation:

```
# Session Log — YYYY-MM-DD HH:MM

## Summary
One-paragraph overview of what was discussed and accomplished.

## Key Decisions & Rationale
- Bullet points of important decisions made and why.

## Work Performed
- Files created or modified (with paths)
- Commands run and their outcomes
- Any debugging or troubleshooting steps

## Open Questions / Next Steps
- Items left unfinished or flagged for future work.
```

Generate the timestamp with: `date +%Y-%m-%d-%H%M%S`

## Step 3: Write Changes Summary → `../summaries/`

Write a Markdown file to `../summaries/` (relative to the working directory, i.e. `/Users/alexmurphy/Desktop/claude_scripts/mapThemVeins/summaries/`).

- **Filename**: `YYYY-MM-DD-HHMMSS-<folder>-changes-summary.md` (same timestamp and folder name as Step 2).
- **Content** — a structured, technical summary:

```
# Changes Summary — YYYY-MM-DD HH:MM

## Commits
List each commit made during this session:
- `<short SHA>` — <commit message>

## Files Modified
For each file changed:
### `path/to/file.py`
- Functions/classes/methods added or changed: `function_name()`, `ClassName.method()`
- Brief description of what changed and why.

## Files Created
For each new file:
### `path/to/new_file.py`
- Purpose of the file
- Key functions/classes defined

## Files Deleted
List any files removed, if applicable.

## Architecture Notes
Any structural or design decisions worth noting for future reference.
```

To get commit SHAs and messages from this session, run `git log --oneline -20` and identify the relevant recent commits.

## Final Output

After all three steps complete, print a brief confirmation:
- Whether the push succeeded
- Paths of the two files written
