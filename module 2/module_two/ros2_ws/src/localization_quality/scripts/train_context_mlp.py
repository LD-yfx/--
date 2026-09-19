#!/usr/bin/env python3
"""Train the scene classifier and context fusion MLP using only NumPy."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


SCENE_FEATURES = (
    "openness",
    "direction_entropy",
    "directional_anisotropy",
    "geometry_complexity",
    "coverage_2d",
    "linearity",
    "planarity",
    "laser_valid_rate",
    "laser_close_ratio",
    "laser_abrupt_change_rate",
    "laser_corridor_degeneracy",
)
FUSION_FEATURES = (
    "q_visual",
    "q_laser",
    "p_corridor",
    "p_hall",
    "p_outdoor",
    "illumination_quality",
    "geometry_complexity",
    "openness",
    "visual_valid",
    "laser_valid",
)
VISUAL_CALIBRATION_FEATURES = (
    "q_visual_raw",
    "illumination_quality",
    "underexposed_ratio",
    "overexposed_ratio",
    "edge_density",
    "trackable_features_norm",
    "feature_match_rate",
    "visual_tracking_ok",
    "visual_localization_covariance_score",
)
LASER_CALIBRATION_FEATURES = (
    "q_laser_raw",
    "laser_valid_rate",
    "laser_close_ratio",
    "coverage_2d",
    "direction_entropy",
    "directional_anisotropy",
    "geometry_complexity",
    "openness",
    "linearity",
)
SCENES = ("corridor", "hall", "outdoor")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Mlp:
    def __init__(self, dimensions, rng):
        self.weights = []
        self.biases = []
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
            scale = np.sqrt(2.0 / input_size)
            self.weights.append(
                rng.normal(0.0, scale, (output_size, input_size)).astype(np.float64)
            )
            self.biases.append(np.zeros((output_size,), dtype=np.float64))

    def forward(self, features):
        activations = [features]
        preactivations = []
        current = features
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            preactivation = current @ weight.T + bias
            preactivations.append(preactivation)
            if layer + 1 == len(self.weights):
                shifted = preactivation - np.max(preactivation, axis=1, keepdims=True)
                exponential = np.exp(shifted)
                current = exponential / np.sum(exponential, axis=1, keepdims=True)
            else:
                current = np.maximum(preactivation, 0.0)
            activations.append(current)
        return current, activations, preactivations

    def gradients(self, features, targets):
        prediction, activations, preactivations = self.forward(features)
        delta = (prediction - targets) / features.shape[0]
        weight_gradients = []
        bias_gradients = []
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients.append(delta.T @ activations[layer])
            bias_gradients.append(np.sum(delta, axis=0))
            if layer:
                delta = delta @ self.weights[layer]
                delta *= preactivations[layer - 1] > 0.0
        weight_gradients.reverse()
        bias_gradients.reverse()
        return prediction, weight_gradients, bias_gradients


class LogisticCalibrator:
    def __init__(self, feature_count):
        self.weight = np.zeros((1, feature_count), dtype=np.float64)
        self.bias = np.zeros((1,), dtype=np.float64)

    def predict(self, features):
        logit = np.clip(features @ self.weight.T + self.bias, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logit))


def train_calibrator(model, features, targets, epochs, learning_rate, seed):
    rng = np.random.RandomState(seed)
    first_weight = np.zeros_like(model.weight)
    second_weight = np.zeros_like(model.weight)
    first_bias = np.zeros_like(model.bias)
    second_bias = np.zeros_like(model.bias)
    step = 0
    indices = np.arange(len(features))
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), 128):
            batch = indices[start:start + 128]
            prediction = model.predict(features[batch])
            delta = (prediction[:, 0] - targets[batch]) / len(batch)
            weight_gradient = delta @ features[batch]
            bias_gradient = np.sum(delta)
            step += 1
            first_weight = 0.9 * first_weight + 0.1 * weight_gradient
            second_weight = (
                0.999 * second_weight + 0.001 * weight_gradient ** 2)
            first_bias = 0.9 * first_bias + 0.1 * bias_gradient
            second_bias = 0.999 * second_bias + 0.001 * bias_gradient ** 2
            model.weight -= learning_rate * (
                first_weight / (1.0 - 0.9 ** step)) / (
                np.sqrt(second_weight / (1.0 - 0.999 ** step)) + 1e-8)
            model.bias -= learning_rate * (
                first_bias / (1.0 - 0.9 ** step)) / (
                np.sqrt(second_bias / (1.0 - 0.999 ** step)) + 1e-8)


def train(
        model, features, targets, epochs, batch_size, learning_rate, seed,
        weight_decay=1e-4, validation_features=None,
        validation_targets=None, patience=60):
    rng = np.random.RandomState(seed)
    parameters = model.weights + model.biases
    first = [np.zeros_like(parameter) for parameter in parameters]
    second = [np.zeros_like(parameter) for parameter in parameters]
    step = 0
    indices = np.arange(features.shape[0])
    best_metric = math.inf
    best_epoch = 0
    best_weights = None
    best_biases = None
    stale_epochs = 0
    trained_epochs = 0
    stopped_early = False
    for epoch in range(epochs):
        trained_epochs = epoch + 1
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            _, weight_gradients, bias_gradients = model.gradients(
                features[batch], targets[batch])
            gradients = weight_gradients + bias_gradients
            step += 1
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                first_hat = first[index] / (1.0 - 0.9 ** step)
                second_hat = second[index] / (1.0 - 0.999 ** step)
                update = first_hat / (np.sqrt(second_hat) + 1e-8)
                if index < len(model.weights):
                    update += weight_decay * parameter
                parameter -= learning_rate * update
        if validation_features is not None and validation_targets is not None:
            validation_prediction = model.forward(validation_features)[0]
            validation_metric = float(np.mean(np.abs(
                validation_prediction - validation_targets)))
            if validation_metric < best_metric - 1e-7:
                best_metric = validation_metric
                best_epoch = epoch + 1
                best_weights = [weight.copy() for weight in model.weights]
                best_biases = [bias.copy() for bias in model.biases]
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                break
    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases
    return {
        "selected_epoch": best_epoch if best_epoch else epochs,
        "trained_epochs": trained_epochs,
        "validation_mae": None if not math.isfinite(best_metric) else best_metric,
        "early_stopped": stopped_early,
        "restored_best_epoch": best_epoch > 0 and best_epoch != trained_epochs,
    }


def read_rows(paths):
    rows = []
    fieldnames = None
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                covariance = float(row.get(
                    "visual_localization_covariance", "nan"))
                tracking_state = row.get("visual_tracking_state", "unknown")
                row["visual_tracking_ok"] = (
                    "1" if tracking_state in ("ok", "tracking") else "0")
                row["visual_localization_covariance_score"] = str(
                    math.exp(-max(0.0, covariance) / 0.25)
                    if math.isfinite(covariance) else 0.0)
                rows.append(row)
    if not rows:
        raise RuntimeError("no training rows were loaded")
    return rows


def require_columns(rows, columns):
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise RuntimeError(f"dataset is missing columns: {', '.join(missing)}")


def matrix(rows, columns):
    return np.asarray([
        [float(row[column]) for column in columns] for row in rows
    ], dtype=np.float64)


def split_rows(rows, split_path):
    if split_path:
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        mapping = {
            run_id: name
            for name, run_ids in split["runs"].items() for run_id in run_ids
        }
        unknown = sorted({row["run_id"] for row in rows} - set(mapping))
        if unknown:
            raise RuntimeError(f"rows reference runs absent from split: {unknown}")
        return {
            name: [row for row in rows if mapping[row["run_id"]] == name]
            for name in ("train", "validation", "test")
        }
    result = {
        name: [row for row in rows if row.get("split") == name]
        for name in ("train", "validation", "test")
    }
    if not all(result.values()):
        raise RuntimeError("provide --split or a split column with train/validation/test")
    return result


def normalize(train_features, other_features):
    mean = np.mean(train_features, axis=0, keepdims=True)
    stddev = np.std(train_features, axis=0, keepdims=True)
    stddev[stddev < 1e-8] = 1.0
    return mean, stddev, [
        (features - mean) / stddev for features in other_features
    ]


def augment(features, targets, seed, copies=4, noise_stddev=0.10):
    """Add small standardized feature noise to improve run-level generalization."""
    rng = np.random.RandomState(seed)
    augmented_features = [features]
    augmented_targets = [targets]
    for _ in range(copies):
        augmented_features.append(
            features + rng.normal(0.0, noise_stddev, features.shape))
        augmented_targets.append(targets)
    return np.concatenate(augmented_features), np.concatenate(augmented_targets)


def one_hot(rows):
    labels = np.asarray([SCENES.index(row["scene"]) for row in rows])
    output = np.zeros((len(rows), len(SCENES)), dtype=np.float64)
    output[np.arange(len(rows)), labels] = 1.0
    return output, labels


def macro_f1(labels, prediction):
    scores = []
    for label in range(len(SCENES)):
        true_positive = np.sum((labels == label) & (prediction == label))
        false_positive = np.sum((labels != label) & (prediction == label))
        false_negative = np.sum((labels == label) & (prediction != label))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        scores.append(2.0 * precision * recall / max(1e-12, precision + recall))
    return float(np.mean(scores))


def opencv_matrix(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    flattened = ", ".join(f"{value:.17g}" for value in values.flat)
    return (
        f"{name}: !!opencv-matrix\n"
        f"   rows: {values.shape[0]}\n"
        f"   cols: {values.shape[1]}\n"
        f"   dt: d\n"
        f"   data: [ {flattened} ]\n"
    )


def write_model(path, scene_model, scene_mean, scene_std,
                fusion_model, fusion_mean, fusion_std,
                visual_calibrator, visual_mean, visual_std,
                laser_calibrator, laser_mean, laser_std):
    content = "%YAML:1.0\n---\nformat_version: 1\n"
    content += opencv_matrix("scene_mean", scene_mean)
    content += opencv_matrix("scene_std", scene_std)
    for index, (weight, bias) in enumerate(
            zip(scene_model.weights, scene_model.biases), start=1):
        content += opencv_matrix(f"scene_weight{index}", weight)
        content += opencv_matrix(f"scene_bias{index}", bias)
    content += opencv_matrix("fusion_mean", fusion_mean)
    content += opencv_matrix("fusion_std", fusion_std)
    for index, (weight, bias) in enumerate(
            zip(fusion_model.weights, fusion_model.biases), start=1):
        content += opencv_matrix(f"fusion_weight{index}", weight)
        content += opencv_matrix(f"fusion_bias{index}", bias)
    content += opencv_matrix("visual_calibration_mean", visual_mean)
    content += opencv_matrix("visual_calibration_std", visual_std)
    content += opencv_matrix(
        "visual_calibration_weight", visual_calibrator.weight)
    content += opencv_matrix(
        "visual_calibration_bias", visual_calibrator.bias)
    content += opencv_matrix("laser_calibration_mean", laser_mean)
    content += opencv_matrix("laser_calibration_std", laser_std)
    content += opencv_matrix(
        "laser_calibration_weight", laser_calibrator.weight)
    content += opencv_matrix(
        "laser_calibration_bias", laser_calibrator.bias)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def pearson(left, right):
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_rank_sum = np.sum(ranks[labels == 1])
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0) /
        (positives * negatives))


def evaluate_localization(rows, weights):
    if "visual_window_error" in rows[0]:
        visual_loss = matrix(rows, ("visual_window_error",))[:, 0]
        laser_loss = matrix(rows, ("laser_window_error",))[:, 0]
    else:
        visual = matrix(
            rows, ("visual_translation_error_m", "visual_rotation_error_rad"))
        laser = matrix(
            rows, ("laser_translation_error_m", "laser_rotation_error_rad"))
        visual_loss = np.sqrt(
            visual[:, 0] ** 2 + 0.25 * visual[:, 1] ** 2)
        laser_loss = np.sqrt(
            laser[:, 0] ** 2 + 0.25 * laser[:, 1] ** 2)
    return float(np.mean(weights[:, 0] * visual_loss + weights[:, 1] * laser_loss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--dataset-kind", choices=("formal", "bootstrap"), default="formal")
    args = parser.parse_args()

    try:
        report_path = Path(args.report).resolve()
        rows = read_rows(args.csv)
        require_columns(
            rows,
            SCENE_FEATURES + FUSION_FEATURES +
            VISUAL_CALIBRATION_FEATURES + LASER_CALIBRATION_FEATURES +
            ("scene", "run_id", "w_visual_star", "w_laser_star",
             "q_visual_star", "q_laser_star",
             "visual_translation_error_m", "visual_rotation_error_rad",
             "laser_translation_error_m", "laser_rotation_error_rad",
             "visual_window_error", "laser_window_error"),
        )
        subsets = split_rows(rows, args.split)

        visual_raw = [
            matrix(subsets[name], VISUAL_CALIBRATION_FEATURES)
            for name in ("train", "validation", "test")
        ]
        visual_mean, visual_std, visual_features = normalize(
            visual_raw[0], visual_raw)
        laser_raw = [
            matrix(subsets[name], LASER_CALIBRATION_FEATURES)
            for name in ("train", "validation", "test")
        ]
        laser_mean, laser_std, laser_features = normalize(
            laser_raw[0], laser_raw)
        visual_calibrator = LogisticCalibrator(
            len(VISUAL_CALIBRATION_FEATURES))
        laser_calibrator = LogisticCalibrator(
            len(LASER_CALIBRATION_FEATURES))
        train_calibrator(
            visual_calibrator, visual_features[0],
            matrix(subsets["train"], ("q_visual_star",))[:, 0],
            500, 0.01, args.seed + 10)
        train_calibrator(
            laser_calibrator, laser_features[0],
            matrix(subsets["train"], ("q_laser_star",))[:, 0],
            500, 0.01, args.seed + 11)
        visual_calibrated = [
            visual_calibrator.predict(features)[:, 0]
            for features in visual_features
        ]
        laser_calibrated = [
            laser_calibrator.predict(features)[:, 0]
            for features in laser_features
        ]

        scene_raw = [matrix(subsets[name], SCENE_FEATURES)
                     for name in ("train", "validation", "test")]
        scene_mean, scene_std, scene_features = normalize(scene_raw[0], scene_raw)
        scene_targets = [one_hot(subsets[name]) for name in (
            "train", "validation", "test")]
        scene_model = Mlp((len(SCENE_FEATURES), 16, 8, 3),
                          np.random.RandomState(args.seed))
        scene_training = train(
            scene_model, scene_features[0], scene_targets[0][0],
            args.epochs, args.batch_size, args.learning_rate, args.seed,
            validation_features=scene_features[1],
            validation_targets=scene_targets[1][0])
        scene_probabilities = [
            scene_model.forward(features)[0] for features in scene_features
        ]

        # Use measured scene probabilities if supplied by the extractor. The
        # final C++ model computes the same fields with the trained classifier.
        fusion_raw = [matrix(subsets[name], FUSION_FEATURES)
                      for name in ("train", "validation", "test")]
        for index in range(3):
            fusion_raw[index][:, 0] = visual_calibrated[index]
            fusion_raw[index][:, 1] = laser_calibrated[index]
            fusion_raw[index][:, 2:5] = scene_probabilities[index]
        fusion_mean, fusion_std, fusion_features = normalize(fusion_raw[0], fusion_raw)
        fusion_targets = [
            matrix(subsets[name], ("w_visual_star", "w_laser_star"))
            for name in ("train", "validation", "test")
        ]
        fusion_model = Mlp((len(FUSION_FEATURES), 32, 16, 2),
                           np.random.RandomState(args.seed + 1))
        augmented_features, augmented_targets = augment(
            fusion_features[0], fusion_targets[0], args.seed + 101)
        fusion_training = train(
            fusion_model, augmented_features, augmented_targets,
            args.epochs, args.batch_size, args.learning_rate, args.seed + 1,
            validation_features=fusion_features[1],
            validation_targets=fusion_targets[1])

        output_path = Path(args.output)
        write_model(
            output_path, scene_model, scene_mean, scene_std,
            fusion_model, fusion_mean, fusion_std,
            visual_calibrator, visual_mean, visual_std,
            laser_calibrator, laser_mean, laser_std)

        metrics = {}
        for index, name in enumerate(("train", "validation", "test")):
            scene_prediction = scene_probabilities[index]
            fusion_prediction = fusion_model.forward(fusion_features[index])[0]
            labels = scene_targets[index][1]
            q_visual = visual_calibrated[index]
            q_laser = laser_calibrated[index]
            visual_error = matrix(
                subsets[name], ("visual_window_error",))[:, 0]
            laser_error = matrix(
                subsets[name], ("laser_window_error",))[:, 0]
            failure_labels = np.minimum(
                matrix(subsets[name], ("q_visual_star",))[:, 0],
                matrix(subsets[name], ("q_laser_star",))[:, 0]) < 0.30
            failure_score = 1.0 - np.minimum(q_visual, q_laser)
            fixed = np.full((len(subsets[name]), 2), 0.5)
            # The rule baseline is defined only from the extractor's
            # per-frame heuristic scores.  The q_visual/q_laser columns may
            # already contain values from a deployed calibrator and therefore
            # must never be used as training baselines.
            rule = matrix(
                subsets[name], ("q_visual_raw", "q_laser_raw"))
            rule /= np.maximum(np.sum(rule, axis=1, keepdims=True), 1e-12)
            metrics[name] = {
                "rows": len(subsets[name]),
                "scene_macro_f1": macro_f1(
                    labels, np.argmax(scene_prediction, axis=1)),
                "weight_mae": float(np.mean(np.abs(
                    fusion_prediction - fusion_targets[index]))),
                "q_visual_negative_error_correlation": pearson(
                    q_visual, -visual_error),
                "q_laser_negative_error_correlation": pearson(
                    q_laser, -laser_error),
                "failure_auroc": auroc(failure_labels, failure_score),
                "localization_loss_fixed": evaluate_localization(
                    subsets[name], fixed),
                "localization_loss_rule": evaluate_localization(
                    subsets[name], rule),
                "localization_loss_full_context": evaluate_localization(
                    subsets[name], fusion_prediction),
            }
            rule_loss = metrics[name]["localization_loss_rule"]
            model_loss = metrics[name]["localization_loss_full_context"]
            metrics[name]["improvement_over_rule"] = (
                (rule_loss - model_loss) / rule_loss if rule_loss > 0.0 else 0.0)

        model_sha = sha256_file(output_path)
        dataset_inputs = []
        for csv_path in args.csv:
            path = Path(csv_path).resolve()
            dataset_inputs.append({
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        split_provenance = None
        if args.split:
            split_path = Path(args.split).resolve()
            split_provenance = {
                "path": split_path.name,
                "bytes": split_path.stat().st_size,
                "sha256": sha256_file(split_path),
            }
        report = {
            "format_version": 1,
            "algorithm": "numpy_mlp_adam",
            "dataset_kind": args.dataset_kind,
            "seed": args.seed,
            "epochs": args.epochs,
            "training_augmentation": {
                "copies": 4,
                "standardized_gaussian_noise": 0.10,
            },
            "training_hyperparameters": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "maximum_epochs": args.epochs,
            },
            "model_selection": {
                "criterion": "validation_mae",
                "patience": 60,
                "scene": scene_training,
                "fusion": fusion_training,
            },
            "scene_features": SCENE_FEATURES,
            "fusion_features": FUSION_FEATURES,
            "visual_calibration_features": VISUAL_CALIBRATION_FEATURES,
            "laser_calibration_features": LASER_CALIBRATION_FEATURES,
            "scene_architecture": [len(SCENE_FEATURES), 16, 8, 3],
            "fusion_architecture": [len(FUSION_FEATURES), 32, 16, 2],
            "quality_contract": {
                "raw_scores": ["q_visual_raw", "q_laser_raw"],
                "calibrated_scores": [
                    "visual_calibrator_output",
                    "laser_calibrator_output",
                ],
                "rule_baseline_features": [
                    "q_visual_raw", "q_laser_raw",
                ],
                "fusion_model_features": list(FUSION_FEATURES),
            },
            "dataset_inputs": dataset_inputs,
            "split_file": split_provenance,
            "model_path": Path(
                os.path.relpath(
                    output_path.resolve(), start=report_path.parent.resolve())
            ).as_posix(),
            "model_sha256": model_sha,
            "metrics": metrics,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"train_context_mlp: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
