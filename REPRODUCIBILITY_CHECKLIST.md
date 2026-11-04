# Reproducibility Checklist

Use this checklist before sharing a CBIR experiment or reporting retrieval results.
Unchecked items are pending verification, not completed checks.

## Environment

- [ ] Record the Git commit used for the experiment (`git rev-parse HEAD`).
- [ ] Record the Python version (`python --version`).
- [ ] Install dependencies in an isolated environment using `python -m pip install -r requirements.txt`.
- [ ] Run `python -m pip check` and investigate dependency conflicts.
- [ ] Save installed package versions using `python -m pip freeze > environment.txt`.
- [ ] Record the operating system, GPU model, and accelerator/runtime versions when applicable.

`pip check` checks package dependency metadata. It does not verify GPU compatibility,
successful model loading, or correct retrieval behavior. A package snapshot is an
environment record, not a guarantee of portability across machines.

## Data and Model

- [ ] Document the dataset source, version, license, and directory structure.
- [ ] Record query and gallery sizes and the rule used to define a relevant result.
- [ ] Check for duplicate images and unintended overlap between evaluation and tuning data.
- [ ] Explicitly state whether a query image is also in the gallery and whether self-matches are excluded.
- [ ] Record the exact backbone/checkpoint identifier and revision when available.
- [ ] Document image resizing, cropping, color conversion, and normalization.
- [ ] Record how features are extracted, pooled, and optionally normalized.
- [ ] Document the similarity or distance function and whether higher or lower values rank first.

## Retrieval Evaluation

- [ ] Verify that feature rows still match their original image paths and IDs.
- [ ] Check for unreadable images, missing embeddings, and non-finite feature values.
- [ ] State the chosen values of K and how queries with no relevant gallery items are handled.
- [ ] Define the metrics actually used, including the denominator for precision or recall.
- [ ] Keep query/gallery data, preprocessing, and evaluation rules fixed across model comparisons.
- [ ] Inspect representative successes and failures, not only the best-looking queries.
- [ ] Separate embedding extraction time from retrieval time if reporting performance.
- [ ] Report hardware, gallery size, batch size, and timing procedure with speed measurements.

## Notebook and Artifacts

- [ ] Run the notebook from a fresh session in cell order.
- [ ] Confirm that required outputs exist and no exception outputs remain.
- [ ] Save configuration, metrics, and a small retrieval example set with image IDs.
- [ ] Explain any steps that need Internet access, credentials, or externally downloaded weights.
- [ ] Exclude credentials, private data, and large model/index files from Git.
- [ ] Link to large artifacts separately, with access instructions and checksums where practical.

## Run Record

Fill this section with observed values after running the experiment.

| Item | Observed value |
| --- | --- |
| Run date | Pending |
| Git commit | Pending |
| Python and dependency snapshot | Pending |
| Hardware | Pending |
| Dataset version and split | Pending |
| Query / gallery counts | Pending |
| Model checkpoint | Pending |
| Similarity metric | Pending |
| Evaluation metrics and K | Pending |
| Output artifact location | Pending |
| Known limitations | Pending |
