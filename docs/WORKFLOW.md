# EmberEye Development Workflow

**Purpose:** a simple workflow for one developer working with Claude Code. `main` always works, nothing sensitive reaches the public repo, and every change is small enough to review.
**Last updated:** 2026-10-24

---

## 1. Branches and pull requests

- **One branch per change**, named by type: `feat/georef-flat`, `fix/...`, `docs/...`, `chore/...`, `exp/...` (experiments).
- **Small PRs:** one idea each, under about 200 changed lines where possible.
- **Before opening a PR:** `git fetch origin`, then `git rebase origin/main`.
- **Commit messages** follow Conventional Commits, `type(scope): summary`, and mention the issue, e.g. `feat(georef): rays to flat ground (#12)`.
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, `ci`, `data`, `exp`.
- **Merge with "Rebase and merge"**, so history stays linear and every commit is kept.
- **Never force-push `main`.** Force-pushing your own branch after a rebase is fine: `git push --force-with-lease`.
- **Commit email:** commits use the WKS-DS noreply address (set in this repo's git config). They count on the contribution graph without exposing a real email.

---

## 2. Checks

**On every commit (pre-commit).** One-time setup is `pre-commit install`. To run every check over the whole repo: `pre-commit run --all-files`.
- **ruff:** lint. Formatting follows in its own PR.
- **gitleaks:** blocks anything that looks like a key or token.
- **nbstripout:** strips notebook outputs, so no images, data previews or file paths get committed.
- **Large-file check:** blocks files over 5 MB.

**On every pull request (GitHub Actions, `.github/workflows/ci.yml`).** It runs `ruff check` and `pytest` on Python 3.11 (Ubuntu). MuJoCo renders in software (OSMesa), since the runner has no screen.

---

## 3. Secrets and data

- **Secrets** live in `.env`, which is gitignored. `.env.example` lists their names with placeholders, and code reads them from the environment.
- **Datasets** stay in `data/`, which is gitignored. `data/README.md` says where each one comes from and its licence. Never commit them.
- **Partner data** (anything shared by agencies or companies) lives outside the repo, e.g. in `~/EmberEye-private/`. It never touches git.
- **Model weights** are not in git. Attach them to GitHub Releases.
- **Notebooks** have their outputs stripped automatically. Finished logic moves into `src/`, with tests.

---

## 4. Issues

Keep a plain list of GitHub issues, one per task. Close each one from the PR that does it by writing "Closes #12" in the PR description. No project board, labels or milestones.

---

## 5. GitHub settings (Will, once)

- [ ] **Branch protection on `main`:** require a pull request, require the CI check to pass, and block force-pushes and deletion.
- [ ] **Secret scanning and push protection** (free for public repos).
- [ ] **Dependabot alerts.**
- [ ] **Email privacy:** "Keep my email addresses private" and "Block command line pushes that expose my email". The repo already commits with the noreply address.
- [ ] **Licence:** choose one (MIT, or Apache-2.0 for its patent grant) and add `LICENSE`. Datasets keep their own licences (`data/README.md`).
