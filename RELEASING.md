# Releasing rag-titan

This guide is for maintainers. It is kept out of the public docs site. Releases use *trusted publishing*, so no API
token is stored in GitHub.

## How it works

| You do                                            | The **Publish** workflow does         |
|---------------------------------------------------|---------------------------------------|
| Actions → Publish → **Run workflow**              | test, build, upload to **TestPyPI**   |
| Publish a **GitHub Release** (tag `v0.1.0`)       | test, build, **TestPyPI**, then **PyPI** |

## One-time setup

On **both** [PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/), add a
*pending publisher* with these values:

| Field             | PyPI           | TestPyPI       |
|-------------------|----------------|----------------|
| PyPI project name | `rag-titan`    | `rag-titan`    |
| Owner             | `aion7891`     | `aion7891`     |
| Repository name   | `rag-titan`    | `rag-titan`    |
| Workflow name     | `publish.yml`  | `publish.yml`  |
| Environment name  | `pypi`         | `testpypi`     |

> [!WARNING]
> The values must match exactly. If the environment name on PyPI is different
> from the one in `.github/workflows/publish.yml`, the upload fails with
> `invalid-publisher`.

GitHub creates the `pypi` and `testpypi` environments the first time the
workflow runs. To ask for approval before the real upload, open
**Settings → Environments → pypi** and add yourself as a *required reviewer*.

## Release steps

1. Update `__version__` in `src/rag_titan/__init__.py` (for example `0.1.1`).
2. Add the changes to `CHANGELOG.md`.
3. Merge to `main`.
4. (Optional) Run the **Publish** workflow by hand and check the package on TestPyPI:
   ```bash
   pip install -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ rag-titan
   ```
5. Only **after** the merge, update `main` and check the version:
   ```bash
   git checkout main && git pull
   grep __version__ src/rag_titan/__init__.py    # must show the new version
   ```
6. Create the **Release** from `main` (this also creates the tag):
   ```bash
   gh release create v0.1.1 --target main --title "0.1.1" --notes-file notes.md
   ```
   The notes are the text shown on the Release page; use the new CHANGELOG section.
7. Check that it really published: https://pypi.org/project/rag-titan/ shows
   the new version, and `pip install "rag-titan==0.1.1"` works. PyPI can take a
   few minutes to show it everywhere, and the README badge up to a few hours.

> [!NOTE]
> PyPI and TestPyPI never accept the same version twice, even after deleting it.
> Always increase `__version__` for a new upload.

### The release guard

For releases, the Publish workflow first checks the version and stops with a
clear error if:

- the tag does not match `__version__` (for example a release created before
  the PR was merged, which would rebuild the old version), or
- that version is already on PyPI (`__version__` was not bumped).

If it stops, nothing was uploaded. Delete the release and its tag
(`gh release delete v0.1.1 --cleanup-tag --yes`), fix the cause, and create the
release again.

## Building and checking locally

```bash
pip install -e ".[dev]"
pytest
python -m build              # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
```

## Docs website

The **Docs** workflow publishes this site to GitHub Pages on every push to `main`.
Enable it once in **Settings → Pages → Source: GitHub Actions**.
