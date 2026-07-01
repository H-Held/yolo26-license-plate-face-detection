# Release guide — shipping the trained model

The public repo ships **code + the trained weights only** — never images, annotations, tokens,
or any hint of which external datasets were used.

## 1. Pick the weights
`data_root/runs/faces/<run_name>/weights/best.pt`. Copy it to the repo's `models/` under the
stable filename the release workflow expects (kept constant across versions so download links
don't break). Weights are tracked with Git LFS (see `.gitattributes`).

## 2. Evaluate honestly
Run the model on the held-out **test** split and record precision/recall per class.
- If a class's test set includes lots of external data of a different nature than your own
  photos (e.g. crowd faces), its recall may look low against your target. **Document that it is
  expected and why — do not "fix" it by altering the test split.**

## 3. Scrub every deliverable of hidden-dataset traces
Before committing README / MODEL_CARD / metrics:
- No dataset **names, URLs, or paths** for any hidden source.
- No presence-file semantics that reveal a filtered external set.
- No tunnel/export links, tokens, or server paths.
- Metrics rows must not include a row that only makes sense if a named external set was used.

Quick check from the repo root (must print nothing). Put the real names/URLs of your
private sources in `../.hiden/scrub_terms.txt` (one regex per line, git-ignored) and run:
```
grep -rniE -f ../.hiden/scrub_terms.txt \
    --include=*.md --include=*.py --include=*.yaml .   # .hiden itself is git-ignored
```

## 4. Confirm nothing private is staged
```
git status --porcelain | grep -E "\.hiden|_fixtures|\.jpg|\.zip|kombiniert|_gt\.txt|boxes\.json" \
  && echo "STOP: private/data files staged" || echo "clean"
```

## 5. Tag + release
Follow the repo's existing release workflow (stable model filename, version tag, GitHub release).
Keep release notes clean — no tool footers, no external-dataset mentions.
