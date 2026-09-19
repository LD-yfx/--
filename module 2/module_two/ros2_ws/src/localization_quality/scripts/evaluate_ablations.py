#!/usr/bin/env python3
"""Train and compare the required fixed, rule, context, and ablation baselines."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import train_context_mlp as training


VARIANTS = {
    "sensor_quality_mlp": (
        "q_visual", "q_laser", "visual_valid", "laser_valid",
    ),
    "full_context_mlp": training.FUSION_FEATURES,
    "lighting_geometry_context": tuple(
        name for name in training.FUSION_FEATURES if not name.startswith("p_")
    ),
    "scene_geometry_context": tuple(
        name for name in training.FUSION_FEATURES
        if name != "illumination_quality"
    ),
    "scene_lighting_context": tuple(
        name for name in training.FUSION_FEATURES
        if name != "geometry_complexity"
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    try:
        rows = training.read_rows(args.csv)
        subsets = training.split_rows(rows, args.split)
        test_rows = subsets["test"]
        visual_raw = [
            training.matrix(subsets[name], training.VISUAL_CALIBRATION_FEATURES)
            for name in ("train", "validation", "test")
        ]
        visual_mean, visual_std, visual_features = training.normalize(
            visual_raw[0], visual_raw)
        laser_raw = [
            training.matrix(subsets[name], training.LASER_CALIBRATION_FEATURES)
            for name in ("train", "validation", "test")
        ]
        laser_mean, laser_std, laser_features = training.normalize(
            laser_raw[0], laser_raw)
        visual_calibrator = training.LogisticCalibrator(
            len(training.VISUAL_CALIBRATION_FEATURES))
        laser_calibrator = training.LogisticCalibrator(
            len(training.LASER_CALIBRATION_FEATURES))
        training.train_calibrator(
            visual_calibrator, visual_features[0],
            training.matrix(
                subsets["train"], ("q_visual_star",))[:, 0],
            500, 0.01, args.seed + 10)
        training.train_calibrator(
            laser_calibrator, laser_features[0],
            training.matrix(
                subsets["train"], ("q_laser_star",))[:, 0],
            500, 0.01, args.seed + 11)
        visual_calibrated = [
            visual_calibrator.predict(features)[:, 0]
            for features in visual_features
        ]
        laser_calibrated = [
            laser_calibrator.predict(features)[:, 0]
            for features in laser_features
        ]
        scene_raw = [
            training.matrix(subsets[name], training.SCENE_FEATURES)
            for name in ("train", "validation", "test")
        ]
        _, _, scene_features = training.normalize(scene_raw[0], scene_raw)
        scene_targets = [
            training.one_hot(subsets[name])[0]
            for name in ("train", "validation", "test")
        ]
        scene_model = training.Mlp(
            (len(training.SCENE_FEATURES), 16, 8, 3),
            np.random.RandomState(args.seed))
        scene_training = training.train(
            scene_model, scene_features[0], scene_targets[0],
            args.epochs, 128, args.learning_rate, args.seed,
            validation_features=scene_features[1],
            validation_targets=scene_targets[1])
        scene_probabilities = [
            scene_model.forward(features)[0] for features in scene_features
        ]
        q_visual = training.matrix(test_rows, ("q_visual_raw",))[:, 0]
        q_laser = training.matrix(test_rows, ("q_laser_raw",))[:, 0]
        fixed_weights = np.full((len(test_rows), 2), 0.5)
        rule_weights = np.column_stack((q_visual, q_laser))
        zero = np.sum(rule_weights, axis=1) <= 1e-12
        rule_weights[zero] = 0.5
        rule_weights[~zero] /= np.sum(
            rule_weights[~zero], axis=1, keepdims=True)

        results = {
            "fixed_0_5": {
                "localization_loss": training.evaluate_localization(
                    test_rows, fixed_weights),
            },
            "quality_ratio_rule": {
                "localization_loss": training.evaluate_localization(
                    test_rows, rule_weights),
            },
        }
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        targets = {
            name: training.matrix(subsets[name], (
                "w_visual_star", "w_laser_star"))
            for name in ("train", "validation", "test")
        }
        for offset, (variant, feature_names) in enumerate(VARIANTS.items()):
            raw = [
                training.matrix(subsets[name], feature_names)
                for name in ("train", "validation", "test")
            ]
            for subset_index in range(3):
                if "q_visual" in feature_names:
                    raw[subset_index][:, feature_names.index("q_visual")] = (
                        visual_calibrated[subset_index])
                if "q_laser" in feature_names:
                    raw[subset_index][:, feature_names.index("q_laser")] = (
                        laser_calibrated[subset_index])
                for probability_index, probability_name in enumerate(
                        ("p_corridor", "p_hall", "p_outdoor")):
                    if probability_name in feature_names:
                        raw[subset_index][:, feature_names.index(
                            probability_name)] = scene_probabilities[
                                subset_index][:, probability_index]
            mean, stddev, features = training.normalize(raw[0], raw)
            model = training.Mlp(
                (len(feature_names), 32, 16, 2),
                np.random.RandomState(args.seed + offset))
            augmented_features, augmented_targets = training.augment(
                features[0], targets["train"], args.seed + 100 + offset)
            selection = training.train(
                model, augmented_features, augmented_targets, args.epochs,
                128, args.learning_rate, args.seed + offset,
                validation_features=features[1],
                validation_targets=targets["validation"])
            prediction = model.forward(features[2])[0]
            loss = training.evaluate_localization(test_rows, prediction)
            results[variant] = {
                "feature_names": feature_names,
                "weight_mae": float(np.mean(np.abs(
                    prediction - targets["test"]))),
                "localization_loss": loss,
                "model_selection": selection,
            }
            arrays = {"mean": mean, "stddev": stddev}
            for layer, (weight, bias) in enumerate(
                    zip(model.weights, model.biases), start=1):
                arrays[f"weight{layer}"] = weight
                arrays[f"bias{layer}"] = bias
            np.savez_compressed(model_dir / f"{variant}.npz", **arrays)

        rule_loss = results["quality_ratio_rule"]["localization_loss"]
        for result in results.values():
            result["improvement_over_rule"] = (
                (rule_loss - result["localization_loss"]) / rule_loss
                if rule_loss > 0.0 else 0.0)
        report = {
            "format_version": 1,
            "split": "complete_runs_only",
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "quality_contract": {
                "rule_baseline_features": [
                    "q_visual_raw", "q_laser_raw",
                ],
                "learned_fusion_features": list(training.FUSION_FEATURES),
            },
            "scene_model_selection": scene_training,
            "results": results,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"evaluate_ablations: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
