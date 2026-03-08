# Folder Publish Checklist

Use this checklist before every folder-only GitHub publish.

## A. Scope and intent

- [ ] User clearly specified the folder to publish.
- [ ] Out-of-scope folders are explicitly excluded.
- [ ] Root metadata files allowed for publish are defined.

## B. Access and target validation

- [ ] GitHub owner and repository name are verified.
- [ ] SSH key or token authentication works.
- [ ] Remote repository exists (or create it first).

## C. Clean packaging

- [ ] Temporary clean publish repo is created.
- [ ] Only target folder plus approved metadata are copied.
- [ ] No temporary artifacts or cache directories are included.

## D. Push strategy

- [ ] Normal push if remote is empty or fast-forward compatible.
- [ ] Merge if remote history must be preserved.
- [ ] Force push only when folder-only cleanup is required.

## E. Post-push checks

- [ ] Remote root contents match intended scope.
- [ ] Latest commit message is clear and professional.
- [ ] User receives repo URL and final verification summary.
