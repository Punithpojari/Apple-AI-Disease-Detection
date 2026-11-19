# AppleCare AI model training and evaluation

The candidate was selected using validation macro F1 only. The held-out test set was evaluated after selection and was not used for tuning.

## Dataset

- Source: `/Users/punithc/Downloads/processed_dataset (5)`
- Excluded conflicting source IDs: 12
- Unreadable images: 0
- Remaining class labels are not independently expert-verified.

| Split | APPLE ROT LEAVES | HEALTHY LEAVES | LEAF BLOTCH | SCAB LEAVES | Total |
|---|---:|---:|---:|---:|---:|
| train | 66 | 29 | 76 | 104 | 275 |
| val | 14 | 7 | 16 | 23 | 60 |
| test | 14 | 7 | 16 | 23 | 60 |

## Validation model selection

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Original MobileNetV2 | 0.583 | 0.557 | 0.574 |
| Fine-tuned MobileNetV2 | 0.483 | 0.484 | 0.481 |
| Fine-tuned EfficientNetB0 | 0.567 | 0.566 | 0.567 |

## Training setup

- Selected backbone: pretrained ImageNet `EfficientNetB0`.
- Head: Global Average Pooling, Dropout(0.3), Dense(128, ReLU), Dropout(0.3), and a four-class softmax output.
- Training: frozen-backbone head training followed by upper-layer fine-tuning; Adam, ReduceLROnPlateau, early stopping, checkpointing, and balanced class weights.
- Train-only online augmentation: horizontal flips, small rotations, and moderate brightness changes.
- The EfficientNet candidate accepts 0–255 float pixels and uses its built-in ImageNet preprocessing; Flask sends the same pixel scale.

Validation-selected candidate: `EfficientNetB0`.
Promotion recommendation: yes; EfficientNetB0 was promoted after validation-based selection and clean-test comparison.

## Original-versus-candidate test comparison

- Accuracy: 0.583 → 0.667 (+0.083).
- Macro F1: 0.593 → 0.680.
- Weighted F1: 0.564 → 0.663.

## Original model — clean validation

- Images: 60
- Accuracy: 0.583
- Macro F1: 0.557
- Weighted F1: 0.574

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| APPLE ROT LEAVES | 0.556 | 0.357 | 0.435 | 14 |
| HEALTHY LEAVES | 0.444 | 0.571 | 0.500 | 7 |
| LEAF BLOTCH | 0.619 | 0.812 | 0.703 | 16 |
| SCAB LEAVES | 0.619 | 0.565 | 0.591 | 23 |

Confusion matrix (rows = actual, columns = predicted; order: APPLE ROT LEAVES, HEALTHY LEAVES, LEAF BLOTCH, SCAB LEAVES):

```text
[[5, 2, 3, 4], [1, 4, 0, 2], [1, 0, 13, 2], [2, 3, 5, 13]]
```

## Selected candidate — clean validation

- Images: 60
- Accuracy: 0.567
- Macro F1: 0.566
- Weighted F1: 0.567

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| APPLE ROT LEAVES | 0.615 | 0.571 | 0.593 | 14 |
| HEALTHY LEAVES | 0.455 | 0.714 | 0.556 | 7 |
| LEAF BLOTCH | 0.615 | 0.500 | 0.552 | 16 |
| SCAB LEAVES | 0.565 | 0.565 | 0.565 | 23 |

Confusion matrix (rows = actual, columns = predicted; order: APPLE ROT LEAVES, HEALTHY LEAVES, LEAF BLOTCH, SCAB LEAVES):

```text
[[8, 2, 1, 3], [1, 5, 0, 1], [1, 1, 8, 6], [3, 3, 4, 13]]
```

## Original model — clean test

- Images: 60
- Accuracy: 0.583
- Accuracy 95% Wilson interval: [0.457, 0.699]
- Macro F1: 0.593
- Weighted F1: 0.564

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| APPLE ROT LEAVES | 0.615 | 0.571 | 0.593 | 14 |
| HEALTHY LEAVES | 0.545 | 0.857 | 0.667 | 7 |
| LEAF BLOTCH | 0.565 | 0.812 | 0.667 | 16 |
| SCAB LEAVES | 0.615 | 0.348 | 0.444 | 23 |

Confusion matrix (rows = actual, columns = predicted; order: APPLE ROT LEAVES, HEALTHY LEAVES, LEAF BLOTCH, SCAB LEAVES):

```text
[[8, 0, 3, 3], [1, 6, 0, 0], [1, 0, 13, 2], [3, 5, 7, 8]]
```

## Selected candidate — clean test

- Images: 60
- Accuracy: 0.667
- Accuracy 95% Wilson interval: [0.541, 0.773]
- Macro F1: 0.680
- Weighted F1: 0.663

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| APPLE ROT LEAVES | 0.750 | 0.643 | 0.692 | 14 |
| HEALTHY LEAVES | 0.583 | 1.000 | 0.737 | 7 |
| LEAF BLOTCH | 0.688 | 0.688 | 0.688 | 16 |
| SCAB LEAVES | 0.650 | 0.565 | 0.605 | 23 |

Confusion matrix (rows = actual, columns = predicted; order: APPLE ROT LEAVES, HEALTHY LEAVES, LEAF BLOTCH, SCAB LEAVES):

```text
[[9, 0, 2, 3], [0, 7, 0, 0], [0, 1, 11, 4], [3, 4, 3, 13]]
```

## Limitations

On the cleaned test split, the selected candidate's accuracy is compared with the original model above. The test set contains only 60 images; accuracy intervals overlap, so the observed difference does not establish a robust generalization gain. The supplied archive contains processed splits only, class imbalance remains, and source/plant provenance is unavailable. Remaining labels are not expert-verified. The model score is not a calibrated probability and this classifier does not establish that an input is an apple leaf or recognize diseases outside its four labels.
