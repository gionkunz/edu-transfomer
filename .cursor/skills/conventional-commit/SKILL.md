---
name: conventional-commit
description: Create conventional git commits by analyzing staged and unstaged changes, determining commit type (feat, fix, docs, refactor, etc.), detecting breaking changes, and generating a well-structured commit message. Use when the user asks to commit, create a commit, or make a conventional commit.
---

# Conventional Commit

Create a conventional commit by analyzing the current git changes and generating a structured commit message.

## Workflow

1. **Inspect changes**

Run these commands in parallel:

```bash
git status
git diff
git diff --cached
git log --oneline -5
```

2. **Classify the commit type**

Determine the primary type from the changes:

| Type       | When to use                                       |
| ---------- | ------------------------------------------------- |
| `feat`     | New feature or capability visible to the user     |
| `fix`      | Bug fix                                           |
| `docs`     | Documentation only                                |
| `refactor` | Code restructuring with no behaviour change       |
| `test`     | Adding or updating tests                          |
| `chore`    | Build, CI, tooling, dependency updates            |
| `style`    | Formatting, whitespace, linting (no logic change) |
| `perf`     | Performance improvement                           |
| `ci`       | CI/CD pipeline changes                            |
| `build`    | Build system or external dependency changes       |
| `revert`   | Reverts a previous commit                         |

3. **Determine scope**

Derive the scope from the primary area of change. Use the Nx library or app name when the change is localised (e.g. `declaration-flow`, `api-gateway`, `shared-utils`). Omit the scope if the change spans many areas.

4. **Detect breaking changes**

A change is **breaking** if it:

- Removes or renames a public API, endpoint, or exported symbol
- Changes a function signature in a shared library
- Alters database schema in a non-additive way
- Removes or renames environment variables or config keys
- Changes the behaviour of an existing feature in a way that requires consumer updates

If breaking, append `!` after the scope and add a `BREAKING CHANGE:` footer.

5. **Draft the commit message**

Format:

```
<type>(<scope>): <short summary>

<optional body — explain *why*, not *what*>

<optional footer(s)>
```

Rules:

- **Subject line**: imperative mood, lowercase, no period, max 72 chars
- **Body**: wrap at 80 chars, explain motivation and context
- **Footer**: `BREAKING CHANGE: <description>` if applicable; reference work items with `Fixes #123` or `Relates to #456` when relevant
- Do **not** commit files that likely contain secrets (`.env`, `credentials.json`, etc.) — warn the user instead
- **NEVER** add `Co-authored-by` trailers or any other attribution that implies AI co-ownership. The user is the sole author — the agent must not claim or share authorship in any form

6. **Present the message to the user for confirmation**

Show the proposed commit message and ask the user to confirm before executing. If the user provides corrections, incorporate them.

7. **Format and validate**

Before committing, run the following commands:

```bash
npx nx format --tui=false
```

If either command modifies files or reports errors, fix them before proceeding.

8. **Execute the commit**

```bash
git add <relevant files>
git commit -m "$(cat <<'EOF'
<commit message>
EOF
)"
git status
```

## Examples

**Feature commit:**

```
feat(declaration-flow): add document scanner with camera capture

Introduce a camera-based document scanner to the declaration flow,
allowing users to capture tax documents directly from mobile devices.
```

**Fix with scope:**

```
fix(api-gateway): prevent duplicate entity resolution jobs

Guard against concurrent resolution triggers by checking for
RUNNING jobs before enqueuing a new one.

Fixes #142
```

**Breaking change:**

```
refactor(shared-utils)!: remove deprecated Canton type alias

Remove the `Canton` type alias that was replaced by
`SwissCanton` in v3. All consumers must update imports.

BREAKING CHANGE: `Canton` type removed from @dexi/shared-utils.
Use `SwissCanton` instead.
```

**Chore (dependencies):**

```
chore: update Angular to v20.3.3 and Nx to v21.6.3
```
