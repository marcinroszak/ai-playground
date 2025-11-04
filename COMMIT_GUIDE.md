# Git Commit Message Format

This repository uses **Conventional Commits** format for commit messages.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Commit Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools

## Examples

```bash
feat: add digit recognition neural network
fix(train): resolve torch import error
docs: update README with setup instructions
refactor(model): simplify architecture
test: add unit tests for prediction
```

## Configuration

The commit message template is configured automatically. When you run `git commit` without `-m`, the template will be shown.

A **commit-msg hook** validates that your commit messages follow this format.

## Usage

1. **Interactive commit** (uses template):
   ```bash
   git commit
   ```

2. **Quick commit** (must follow format):
   ```bash
   git commit -m "feat: add new feature"
   ```

3. **Skip validation** (for merge commits, etc.):
   - The hook automatically allows merge commits
   - Comments starting with `#` are ignored

## Customization

- **Template**: Edit `.gitmessage` to change the template
- **Validation**: Edit `.git/hooks/commit-msg` to modify validation rules
- **Disable template**: `git config --unset commit.template`

