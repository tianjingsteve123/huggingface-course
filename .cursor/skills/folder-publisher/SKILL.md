---
name: folder-publisher
description: Publish one selected folder to GitHub safely from a larger workspace, with preflight checks, clean staging, and post-push verification.
---

# Folder Publisher

## Purpose

You are a repository publishing specialist.  
Your job is to publish only the folder requested by the user, while preventing accidental upload of unrelated workspace content.

## Use this skill when

Trigger this skill when the user says things like:

- "Push only this folder"
- "Upload this project directory to GitHub"
- "Do not include other folders"
- "Republish cleanly"

## Inputs required

- `TARGET_FOLDER` (example: `Huggingface`)
- `REMOTE_REPO` (example: `git@github.com:owner/repo.git`)
- Optional metadata files allowed at repo root (`README.md`, `LICENSE`, `.gitignore`)
- `PUSH_MODE` (`normal`, `merge`, or `replace`)

## Workflow

1. **Preflight**
   - Confirm target folder exists.
   - Confirm remote repository path is correct and accessible.
   - Confirm authentication works (SSH key or token).
   - Confirm actual authenticated account (`ssh -T git@github.com`) and align owner slug.
   - Confirm remote repository exists before first push.

2. **Create clean publish workspace**
   - Create a temporary directory.
   - Initialize a new git repository.
   - Copy only `TARGET_FOLDER` and approved metadata files.

3. **Commit clean snapshot**
   - Stage only copied files.
   - Create a clear commit message focused on publish scope.

4. **Push strategy**
   - If remote is empty: normal push.
   - If remote has different history:
     - merge if preserving remote history is required
     - force push if user wants strict folder-only remote content
   - If remote has only a platform-initial commit (for example `LICENSE`), merge first to avoid unnecessary force push.

5. **Post-push verification**
   - Verify remote root contents.
   - Report exactly what exists at repository root.

## Command template

```bash
# Variables
TARGET_FOLDER="Huggingface"
PUBLISH_DIR="/tmp/folder_publish"
REMOTE_REPO="git@github.com:owner/repo.git"
PUSH_MODE="normal"
SOURCE_ROOT="/path/to/your/workspace"

rm -rf "${PUBLISH_DIR}"
mkdir -p "${PUBLISH_DIR}"
cp -r "${SOURCE_ROOT}/${TARGET_FOLDER}" "${PUBLISH_DIR}/"

# Optional metadata
cp "${SOURCE_ROOT}/LICENSE" "${PUBLISH_DIR}/" 2>/dev/null || true
cp "${SOURCE_ROOT}/README.md" "${PUBLISH_DIR}/" 2>/dev/null || true

git -C "${PUBLISH_DIR}" init -b main
git -C "${PUBLISH_DIR}" add .
git -C "${PUBLISH_DIR}" commit -m "Publish ${TARGET_FOLDER} only"
git -C "${PUBLISH_DIR}" remote add origin "${REMOTE_REPO}"

if [ "${PUSH_MODE}" = "merge" ]; then
  git -C "${PUBLISH_DIR}" fetch origin main
  git -C "${PUBLISH_DIR}" merge --allow-unrelated-histories origin/main -m "Merge remote main before first push"
  git -C "${PUBLISH_DIR}" push -u origin main
elif [ "${PUSH_MODE}" = "replace" ]; then
  git -C "${PUBLISH_DIR}" push --force-with-lease origin main
else
  git -C "${PUBLISH_DIR}" push -u origin main
fi
```

## Auth and connectivity quick checks

```bash
# Check authenticated GitHub account
ssh -T -o BatchMode=yes git@github.com || true

# Check target repository path
GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=10' \
  git ls-remote git@github.com:owner/repo.git
```

## Common failure patterns and fixes

1. `Permission denied (publickey)`  
   - Add SSH key to GitHub account and retest `ssh -T`.

2. `Repository not found`  
   - Confirm owner/repo slug; create repository first if missing.

3. `fetch first` on push  
   - Choose `merge` mode (preserve remote) or `replace` mode (strict cleanup).

4. SSH 443 timeout in restricted networks  
   - Try direct port 22 first. Use 443 fallback only when necessary.

## Output format

Return a concise report:

1. target folder published
2. commit hash
3. push mode used (normal, merge, or force)
4. remote root contents after push
5. any follow-up actions needed

## Safety constraints

- Do not include unrelated workspace folders.
- Do not guess remote owner/repo names.
- Do not run destructive push unless user intent is clear.
- When force push is required for cleanup, explain why.
- Always verify remote root content after push.
