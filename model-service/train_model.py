"""Train and compare pretrained apple-leaf classifiers using clean dataset splits."""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

from audit_dataset import (
    DEFAULT_EXCLUSIONS,
    audit_dataset,
    load_exclusions,
    source_id,
)

SERVICE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = SERVICE_DIR / "classes.json"
DEFAULT_MODEL = SERVICE_DIR / "model" / "apple_disease_efficientnetb0.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 24
SEED = 42
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LR = 1e-5


def load_records(data_root: Path, classes: list[str], excluded: set[str]) -> dict:
    records: dict[str, list[tuple[str, int]]] = {name: [] for name in ("train", "val", "test")}
    for split in records:
        for label, class_name in enumerate(classes):
            folder = data_root / split / class_name
            paths = sorted(path for path in folder.iterdir() if path.is_file())
            if split == "train":
                paths = [path for path in paths if re.search(r"_0\.jpg$", path.name, re.IGNORECASE)]
            for path in paths:
                if source_id(path) not in excluded:
                    records[split].append((str(path), label))
        if not records[split]:
            raise ValueError(f"No usable images remain in the {split!r} split.")
    return records


def validate_clean_splits(audit: dict, excluded: set[str]) -> None:
    remaining_conflicts = [
        item for item in audit["source_id_conflicts"]
        if item["source_id"] not in excluded
    ]
    if remaining_conflicts:
        conflicts = [item["source_id"] for item in remaining_conflicts]
        raise ValueError(
            "Unreviewed source-ID collisions remain. Review the audit report and add "
            f"confirmed duplicates to the exclusion manifest: {conflicts}"
        )
    overlaps = {
        pair: source_ids
        for pair, source_ids in audit["source_overlap_after_exclusions"].items()
        if source_ids
    }
    if overlaps:
        raise ValueError(f"Source images still cross dataset splits: {overlaps}")
    pixel_overlaps = [
        item for item in audit["exact_pixel_duplicate_groups"]
        if len(item["splits"]) > 1
        and any(
            source_id(Path(file["path"])) not in excluded
            for file in item["files"]
        )
    ]
    if pixel_overlaps:
        raise ValueError(
            "Exact pixel duplicates remain across splits after exclusions; "
            "resolve them before training."
        )
    if audit["unreadable_images"]:
        raise ValueError(
            f"Found {len(audit['unreadable_images'])} unreadable images; "
            "see dataset_audit.json."
        )


def make_dataset(records: list[tuple[str, int]], training: bool) -> tf.data.Dataset:
    files = [record[0] for record in records]
    labels = [record[1] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        dataset = dataset.shuffle(
            len(records), seed=SEED, reshuffle_each_iteration=True
        )

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE, antialias=True)
        return tf.cast(image, tf.float32), label

    return (
        dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_model(backbone_name: str, class_count: int) -> tuple[Model, Model]:
    if backbone_name == "mobilenetv2":
        backbone = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SIZE + (3,),
        )
    elif backbone_name == "efficientnetb0":
        backbone = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=IMG_SIZE + (3,),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,), name="image")
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(
                0.06, fill_mode="constant", fill_value=255.0
            ),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
        ],
        name="training_augmentation",
    )
    x = augmentation(inputs)
    if backbone_name == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(class_count, activation="softmax", name="predictions")(x)
    return Model(inputs, outputs, name=f"applecare_{backbone_name}"), backbone


def balanced_class_weights(records: list[tuple[str, int]], class_count: int) -> dict:
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(class_count),
        y=labels,
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def phase_callbacks(checkpoint: tf.keras.callbacks.Callback) -> list:
    return [
        checkpoint,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
    ]


def fit_backbone(
    backbone_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    train_records: list[tuple[str, int]],
    class_count: int,
    output_dir: Path,
) -> tuple[Path, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_checkpoint.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
    )
    model, backbone = build_model(backbone_name, class_count)
    class_weights = balanced_class_weights(train_records, class_count)
    print(
        f"{backbone_name} class weights:",
        {str(key): round(value, 3) for key, value in class_weights.items()},
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n{backbone_name}: stage 1 — training classifier head")
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=phase_callbacks(checkpoint),
        verbose=2,
    )

    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * 0.75)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_at:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(
        f"{backbone_name}: stage 2 — fine-tuning from layer {fine_tune_at} "
        f"of {len(backbone.layers)}"
    )
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=phase_callbacks(checkpoint),
        verbose=2,
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    candidate_path = output_dir / "candidate.keras"
    best_model.save(candidate_path)
    history = {
        "head": head_history.history,
        "fine_tune": fine_history.history,
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    plot_training_history(history, output_dir / "training_curves.png")
    del best_model, model, backbone
    tf.keras.backend.clear_session()
    return candidate_path, history


def score_model(model: Model, dataset: tf.data.Dataset, classes: list[str]) -> dict:
    y_true = np.concatenate([labels.numpy() for _, labels in dataset])
    probabilities = model.predict(dataset, verbose=0)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=np.arange(len(classes)),
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, predictions, average="weighted", zero_division=0
    )
    correct = int(np.sum(y_true == predictions))
    sample_count = len(y_true)
    z = 1.96
    proportion = correct / sample_count
    denominator = 1 + z * z / sample_count
    center = (proportion + z * z / (2 * sample_count)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / sample_count
            + z * z / (4 * sample_count * sample_count)
        )
        / denominator
    )
    return {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "accuracy_95ci_wilson": [
            float(center - margin),
            float(center + margin),
        ],
        "macro": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "weighted": {
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
        },
        "per_class": {
            classes[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(classes))
        },
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=np.arange(len(classes))
        ).tolist(),
    }


def plot_training_history(history: dict, path: Path) -> None:
    values = {}
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        values[key] = history["head"].get(key, []) + history["fine_tune"].get(key, [])
    switch = len(history["head"].get("accuracy", []))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, train_key, val_key, title in (
        (axes[0], "accuracy", "val_accuracy", "Accuracy"),
        (axes[1], "loss", "val_loss", "Loss"),
    ):
        axis.plot(values[train_key], label="train")
        axis.plot(values[val_key], label="validation")
        axis.axvline(switch - 0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_confusion(metrics: dict, classes: list[str], path: Path) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), labels=classes, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), labels=classes)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Clean held-out test confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_report(
    output_dir: Path,
    classes: list[str],
    counts: dict,
    audit: dict,
    baseline_val: dict,
    candidate_val: dict,
    baseline_test: dict,
    candidate_test: dict,
    candidates: dict,
    selected_name: str,
    promotion_recommended: bool,
) -> None:
    display_names = {
        "mobilenetv2": "MobileNetV2",
        "efficientnetb0": "EfficientNetB0",
    }
    lines = [
        "# AppleCare AI model training and evaluation",
        "",
        "The candidate was selected using validation macro F1 only. The held-out test "
        "set was evaluated after selection and was not used for tuning.",
        "",
        "## Dataset",
        "",
        f"- Source: `{audit['data_root']}`",
        f"- Excluded conflicting source IDs: {len(audit['excluded_source_ids'])}",
        f"- Unreadable images: {len(audit['unreadable_images'])}",
        "- Remaining class labels are not independently expert-verified.",
        "",
        "| Split | " + " | ".join(classes) + " | Total |",
        "|---|" + "|".join("---:" for _ in classes) + "|---:|",
    ]
    for split in ("train", "val", "test"):
        split_counts = counts[split]
        lines.append(
            f"| {split} | "
            + " | ".join(str(split_counts[class_name]) for class_name in classes)
            + f" | {sum(split_counts.values())} |"
        )
    lines.extend(
        [
            "",
            "## Validation model selection",
            "",
            "| Model | Accuracy | Macro F1 | Weighted F1 |",
            "|---|---:|---:|---:|",
            f"| Original MobileNetV2 | {baseline_val['accuracy']:.3f} | "
            f"{baseline_val['macro']['f1']:.3f} | {baseline_val['weighted']['f1']:.3f} |",
        ]
    )
    for backbone_name, result in candidates.items():
        metrics = result["validation"]
        lines.append(
            f"| Fine-tuned {display_names[backbone_name]} | {metrics['accuracy']:.3f} | "
            f"{metrics['macro']['f1']:.3f} | {metrics['weighted']['f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Training setup",
            "",
            f"- Selected backbone: pretrained ImageNet `{display_names[selected_name]}`.",
            "- Head: Global Average Pooling, Dropout(0.3), Dense(128, ReLU), "
            "Dropout(0.3), and a four-class softmax output.",
            "- Training: frozen-backbone head training followed by upper-layer "
            "fine-tuning; Adam, ReduceLROnPlateau, early stopping, checkpointing, "
            "and balanced class weights.",
            "- Train-only online augmentation: horizontal flips, small rotations, "
            "and moderate brightness changes.",
            "- The EfficientNet candidate accepts 0–255 float pixels and uses its "
            "built-in ImageNet preprocessing; Flask sends the same pixel scale.",
        ]
    )

    def add_metrics(title: str, metrics: dict) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"- Images: {metrics['n_images']}",
                f"- Accuracy: {metrics['accuracy']:.3f}",
            ]
        )
        if "accuracy_95ci_wilson" in metrics:
            low, high = metrics["accuracy_95ci_wilson"]
            lines.append(
                f"- Accuracy 95% Wilson interval: [{low:.3f}, {high:.3f}]"
            )
        lines.extend(
            [
                f"- Macro F1: {metrics['macro']['f1']:.3f}",
                f"- Weighted F1: {metrics['weighted']['f1']:.3f}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for class_name, row in metrics["per_class"].items():
            lines.append(
                f"| {class_name} | {row['precision']:.3f} | {row['recall']:.3f} "
                f"| {row['f1']:.3f} | {row['support']} |"
            )
        lines.extend(
            [
                "",
                "Confusion matrix (rows = actual, columns = predicted; order: "
                + ", ".join(classes)
                + "):",
                "",
                "```text",
                json.dumps(metrics["confusion_matrix"]),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            f"Validation-selected candidate: `{display_names[selected_name]}`.",
            "Promotion recommendation: "
            + ("yes" if promotion_recommended else "no"),
            "",
            "## Original-versus-candidate test comparison",
            "",
            f"- Accuracy: {baseline_test['accuracy']:.3f} → "
            f"{candidate_test['accuracy']:.3f} "
            f"({candidate_test['accuracy'] - baseline_test['accuracy']:+.3f}).",
            f"- Macro F1: {baseline_test['macro']['f1']:.3f} → "
            f"{candidate_test['macro']['f1']:.3f}.",
            f"- Weighted F1: {baseline_test['weighted']['f1']:.3f} → "
            f"{candidate_test['weighted']['f1']:.3f}.",
        ]
    )
    add_metrics("Original model — clean validation", baseline_val)
    add_metrics("Selected candidate — clean validation", candidate_val)
    add_metrics("Original model — clean test", baseline_test)
    add_metrics("Selected candidate — clean test", candidate_test)
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "On the cleaned test split, the selected candidate's accuracy is compared "
            "with the original model above. The test set contains only 60 images; "
            "accuracy intervals overlap, so the observed difference does not establish "
            "a robust generalization gain. The supplied archive contains processed "
            "splits only, class imbalance remains, and source/plant provenance is "
            "unavailable. Remaining labels are not expert-verified. The model score is "
            "not a calibrated probability and this classifier does not establish that "
            "an input is an apple leaf or recognize diseases outside its four labels.",
            "",
        ]
    )
    (output_dir / "improved_model_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=SERVICE_DIR / "training_runs" / "latest")
    parser.add_argument("--exclusions-file", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=("mobilenetv2", "efficientnetb0"),
        default=("mobilenetv2", "efficientnetb0"),
    )
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    data_root = args.data_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    excluded = load_exclusions(args.exclusions_file.expanduser().resolve())
    audit = audit_dataset(data_root, classes, excluded)
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    validate_clean_splits(audit, excluded)

    records = load_records(data_root, classes, excluded)
    counts = {
        split: {
            class_name: sum(label == index for _, label in split_records)
            for index, class_name in enumerate(classes)
        }
        for split, split_records in records.items()
    }
    for split, split_counts in counts.items():
        if any(value == 0 for value in split_counts.values()):
            raise ValueError(f"Every class needs examples in {split}: {split_counts}")
    print("Clean source-image counts:", counts)

    datasets = {
        split: make_dataset(split_records, training=(split == "train"))
        for split, split_records in records.items()
    }
    if not model_path.is_file():
        raise FileNotFoundError(f"Original model not found: {model_path}")

    baseline_model = tf.keras.models.load_model(model_path)
    baseline_val = score_model(baseline_model, datasets["val"], classes)
    del baseline_model
    tf.keras.backend.clear_session()
    print(
        "Original clean validation:",
        f"accuracy={baseline_val['accuracy']:.3f}",
        f"macro_f1={baseline_val['macro']['f1']:.3f}",
    )

    candidates: dict[str, dict] = {}
    for backbone_name in args.backbones:
        tf.keras.utils.set_random_seed(SEED)
        run_dir = output_dir / backbone_name
        candidate_path, history = fit_backbone(
            backbone_name,
            datasets["train"],
            datasets["val"],
            records["train"],
            len(classes),
            run_dir,
        )
        candidate_model = tf.keras.models.load_model(candidate_path)
        val_metrics = score_model(candidate_model, datasets["val"], classes)
        val_loss = float(candidate_model.evaluate(datasets["val"], verbose=0)[0])
        candidates[backbone_name] = {
            "path": str(candidate_path),
            "validation": val_metrics,
            "validation_loss": val_loss,
        }
        del candidate_model, history
        tf.keras.backend.clear_session()
        print(
            f"{backbone_name} clean validation:",
            f"accuracy={val_metrics['accuracy']:.3f}",
            f"macro_f1={val_metrics['macro']['f1']:.3f}",
            f"loss={val_loss:.3f}",
        )

    selected_name = max(
        candidates,
        key=lambda name: (
            candidates[name]["validation"]["macro"]["f1"],
            candidates[name]["validation"]["accuracy"],
            -candidates[name]["validation_loss"],
        ),
    )
    selected = candidates[selected_name]
    promotion_recommended = (
        selected["validation"]["macro"]["f1"] > baseline_val["macro"]["f1"]
    )
    print(
        f"Selected by validation macro F1: {selected_name}; "
        f"promotion recommended: {promotion_recommended}"
    )

    baseline_model = tf.keras.models.load_model(model_path)
    baseline_test = score_model(baseline_model, datasets["test"], classes)
    del baseline_model
    tf.keras.backend.clear_session()
    candidate_model = tf.keras.models.load_model(selected["path"])
    candidate_test = score_model(candidate_model, datasets["test"], classes)
    del candidate_model
    tf.keras.backend.clear_session()

    comparison = {
        "selection_metric": "clean validation macro F1",
        "selected_backbone": selected_name,
        "promotion_recommended_by_validation": promotion_recommended,
        "candidate_validation": selected["validation"],
        "baseline_validation": baseline_val,
        "candidate_test": candidate_test,
        "baseline_test": baseline_test,
        "all_candidate_validation": {
            name: {
                "metrics": row["validation"],
                "loss": row["validation_loss"],
                "model_path": row["path"],
            }
            for name, row in candidates.items()
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    plot_confusion(
        candidate_test, classes, output_dir / "confusion_matrix_test.png"
    )
    write_report(
        output_dir,
        classes,
        counts,
        audit,
        baseline_val,
        selected["validation"],
        baseline_test,
        candidate_test,
        candidates,
        selected_name,
        promotion_recommended,
    )
    print(
        "Clean test comparison:",
        f"original accuracy={baseline_test['accuracy']:.3f}, "
        f"macro_f1={baseline_test['macro']['f1']:.3f};",
        f"{selected_name} accuracy={candidate_test['accuracy']:.3f}, "
        f"macro_f1={candidate_test['macro']['f1']:.3f}",
    )
    print(f"Candidate saved separately; original model not modified: {output_dir}")


if __name__ == "__main__":
    main()
