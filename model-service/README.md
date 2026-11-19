# EfficientNetB0 model service

This Flask service loads `model/apple_disease_efficientnetb0.keras` and provides
the four-class inference endpoint used by the AppleCare AI Node backend. The
React app's existing upload, results, dashboard, and history flows are the
user-facing interface; the standalone HTML page from the nested app archive is
not served here.

## Run

Install the requirements with Python 3.11. The supplied model was saved with
Keras 3.15.1 and requires a compatible runtime:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

By default the service listens only on `127.0.0.1:5001`. Configure
`MODEL_HOST` and `MODEL_PORT` when running the Flask app directly. The root
`npm run dev` command starts it automatically and prefers
`model-service/.venv/bin/python` (or its Windows equivalent); set
`MODEL_PYTHON` to override that interpreter.

## Endpoints

- `GET /health` reports the model name and class labels.
- `POST /predict` accepts a multipart `image` and returns the highest-ranked
  label, raw score, uncertainty flag, and scores for all four labels.

Images are EXIF-oriented, converted to RGB, resized to 224×224, then passed as
float pixels to the saved model. The saved model contains the training-time
input rescaling layers. Images smaller than 64×64 are rejected. The service
flags a top score below 0.60 as uncertain; this is a screening threshold, not a
calibrated probability.

## Supplied assets

- `app.py`: local inference service adapted from the supplied model app for the
  existing React/Node application. The archive's separate HTML UI and its
  unsupported per-class explanatory text are not used.
- `classes.json`: original output label order.
- `model/apple_disease_efficientnetb0.keras`: validation-selected EfficientNetB0 candidate.
- `model/backups/apple_disease_mobilenetv2.original.keras`: byte-identical backup
  of the original deployed model.
- `preprocess.py`: raw-image preprocessing, exact duplicate checks, and
  train-only augmentation.
- `audit_dataset.py`: reports unreadable images, duplicate/source-ID conflicts,
  split counts, and possible leakage.
- `train_model.py`: two-stage MobileNetV2 and EfficientNetB0 transfer-learning
  candidates; validation macro F1 selects the candidate, and the test split is
  evaluated only after selection. The result remains separate until explicitly
  promoted.
- `dataset_exclusions.json`: dataset-specific, visually reviewed source IDs
  with conflicting labels; review this manifest before reusing a different
  dataset.
- `evaluate_model.py`: legacy evaluator for the original unfiltered splits.
- `evaluation/metrics_summary.json` and `evaluation/classification_report.txt`:
  supplied evaluation summaries.
- `requirements.txt`: runtime dependencies; `training-requirements.txt` lists
  dependencies for reproducing the training/evaluation workflow.

The original model and its training settings are preserved under
`model/backups/`. The supplied archive contains processed splits but not the
raw source dataset.

## Audit and train on the supplied processed dataset

The uploaded `processed_dataset (5)` folder is already split into `train`,
`val`, and `test` directories. First audit it:

```bash
model-service/.venv/bin/python model-service/audit_dataset.py \
  --data-root "/Users/punithc/Downloads/processed_dataset (5)" \
  --output model-service/evaluation/dataset_audit.json
```

Train both pretrained backbones and compare them with the current model using:

```bash
model-service/.venv/bin/python model-service/train_model.py \
  --data-root "/Users/punithc/Downloads/processed_dataset (5)" \
  --output-dir model-service/training_runs/latest
```

Keras downloads the official ImageNet weights if they are not cached. The
training script uses only `_0.jpg` source copies for training and performs
augmentation online; it excludes all IDs in `dataset_exclusions.json` from all
splits and stops if other source conflicts, cross-split source overlaps,
unreadable images, or exact cross-split pixel duplicates remain. Validation
macro F1 selects the candidate. The original and selected candidate are then
evaluated on the filtered test split; this script does not replace the live
model. Promote a candidate only after reviewing the report.
The output directory contains the dataset audit, candidate models, training
curves, test confusion matrix, comparison JSON, and a Markdown report.

Run the inference route tests with:

```bash
cd model-service
.venv/bin/python -m unittest -v test_app
```

`preprocess.py` expects the original class-folder dataset under
`raw_dataset/APPLE_DISEASE_DATASET`. It rejects unreadable images and identical
images assigned to different classes before splitting. The original source
dataset is not included in the supplied archive, so this preprocessing flow
cannot currently regenerate a new split from raw images.

## Evaluation and limitations

The original report records 55.6% accuracy on 63 held-out images, but the
dataset audit found visually duplicated images with conflicting labels across
classes and splits. Treat that result as contaminated, not as a reliable
generalization estimate. The raw source dataset is unavailable, the remaining
labels have not been independently reviewed by an agricultural expert, and the
cleaned held-out test split is small. A model score is not a calibrated
probability. The classifier does not verify apple-leaf identity or recognize
conditions outside its four labels, so results are screening suggestions, not
diagnoses.
