# Folder Publisher Examples

Practical examples for publishing one folder from a larger workspace.

## Example 1: First publish to an empty remote

Use when remote repository is newly created and has no commits.

```bash
TARGET_FOLDER="Huggingface"
PUBLISH_DIR="/tmp/hf_publish_01"
REMOTE_REPO="git@github.com:owner/huggingface-course.git"
SOURCE_ROOT="/path/to/your/workspace"

rm -rf "${PUBLISH_DIR}" && mkdir -p "${PUBLISH_DIR}"
cp -r "${SOURCE_ROOT}/${TARGET_FOLDER}" "${PUBLISH_DIR}/"
cp "${SOURCE_ROOT}/LICENSE" "${PUBLISH_DIR}/" 2>/dev/null || true
cp "${SOURCE_ROOT}/README.md" "${PUBLISH_DIR}/" 2>/dev/null || true

git -C "${PUBLISH_DIR}" init -b main
git -C "${PUBLISH_DIR}" add .
git -C "${PUBLISH_DIR}" commit -m "Publish ${TARGET_FOLDER} only"
git -C "${PUBLISH_DIR}" remote add origin "${REMOTE_REPO}"
git -C "${PUBLISH_DIR}" push -u origin main
```

## Example 2: Remote has initial commit (preserve history)

Use when GitHub remote already has one bootstrap commit (for example `LICENSE`).

```bash
git -C "${PUBLISH_DIR}" fetch origin main
git -C "${PUBLISH_DIR}" merge --allow-unrelated-histories origin/main -m "Merge remote main before first push"
git -C "${PUBLISH_DIR}" push -u origin main
```

## Example 3: Strict scope cleanup (replace remote history)

Use only when user explicitly wants remote to contain target folder only.

```bash
git -C "${PUBLISH_DIR}" push --force-with-lease origin main
```

## Example 4: Verify remote root after push

```bash
curl -s "https://api.github.com/repos/owner/huggingface-course/contents" \
| python3 -c "import sys,json; data=json.load(sys.stdin); print('\n'.join(sorted([x['name'] for x in data])))"
```

Expected output should include only approved entries (for example `Huggingface`, `LICENSE`, `README.md`, `.cursor`).
