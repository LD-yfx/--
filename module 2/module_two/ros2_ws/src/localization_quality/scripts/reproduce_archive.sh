#!/usr/bin/env bash
# Rebuild module one from accepted bags into a self-contained result directory.
set -euo pipefail

project_dir=$(cd "$(dirname "$0")/.." && pwd)
output_dir=${1:-"$project_dir/results/reproduced"}
work_dir=$(mktemp -d /tmp/localization-quality-reproduce.XXXXXX)

cleanup()
{
  rm -rf "$work_dir"
}
trap cleanup EXIT

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "ROS2 Humble is required: /opt/ros/humble/setup.bash not found" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
export ROS_LOG_DIR="$work_dir/ros_logs"
mkdir -p "$ROS_LOG_DIR" "$output_dir"

cmake -S "$project_dir" -B "$work_dir/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$work_dir/install"
cmake --build "$work_dir/build" --parallel 2
cmake --install "$work_dir/build"
source "$work_dir/install/share/localization_quality/local_setup.bash"

python3 "$project_dir/scripts/prepare_dataset.py" validate-matrix \
  "$project_dir/data/formal_gazebo"
python3 "$project_dir/scripts/summarize_formal_dataset.py" \
  "$project_dir/data/formal_gazebo" \
  "$output_dir/formal_dataset_summary.json"
python3 "$project_dir/scripts/build_training_dataset.py" \
  "$project_dir/data/formal_gazebo" "$work_dir/extraction" \
  --params-file "$project_dir/config/quality_params.yaml" \
  --force

cp "$work_dir/extraction/training_rows.csv" "$output_dir/training_rows.csv"
cp "$work_dir/extraction/dataset_split.json" "$output_dir/dataset_split.json"

python3 "$project_dir/scripts/train_context_mlp.py" \
  "$output_dir/training_rows.csv" \
  --split "$output_dir/dataset_split.json" \
  --dataset-kind formal \
  --output "$output_dir/context_mlp_formal.yaml" \
  --report "$output_dir/training_report.json"
python3 "$project_dir/scripts/evaluate_ablations.py" \
  "$output_dir/training_rows.csv" \
  --split "$output_dir/dataset_split.json" \
  --model-dir "$work_dir/ablations" \
  --output "$output_dir/ablation_report.json"

runtime_bag="$project_dir/data/formal_gazebo/corridor_strong_seed0/bag"
for mode in normal laser_fail visual_fail; do
  ros2 run localization_quality offline_evaluator \
    "$runtime_bag" "$mode" "$output_dir/runtime_${mode}.csv" \
    --ros-args \
    --params-file "$project_dir/config/quality_params.yaml" \
    -p "fusion.model_path:=$output_dir/context_mlp_formal.yaml"
done

python3 "$project_dir/scripts/validate_results.py" \
  "$output_dir/runtime_normal.csv" \
  "$output_dir/runtime_laser_fail.csv" \
  "$output_dir/runtime_visual_fail.csv" \
  --output "$output_dir/fault_safety_report.json"
python3 "$project_dir/scripts/generate_formal_quality_maps.py" \
  "$project_dir/data/formal_gazebo" \
  "$output_dir/quality_maps" \
  --split "$output_dir/dataset_split.json" \
  --params-file "$project_dir/config/quality_params.yaml" \
  --model-path "$output_dir/context_mlp_formal.yaml" \
  --summary-json "$output_dir/quality_map_summary.json" \
  --summary-csv "$output_dir/quality_map_summary.csv" \
  --jobs 2
python3 "$project_dir/scripts/validate_quality_map_collection.py" \
  "$output_dir/quality_maps" \
  "$output_dir/quality_map_summary.json" \
  "$output_dir/quality_map_summary.csv" \
  --expected-runs 27

python3 "$project_dir/scripts/validate_module_one.py" \
  --training-report "$output_dir/training_report.json" \
  --runtime-csv "$output_dir/runtime_normal.csv" \
  --ablation-report "$output_dir/ablation_report.json" \
  --quality-map-report "$output_dir/quality_map_summary.json" \
  --expected-sensor-frames 1329 \
  --output "$output_dir/acceptance_report.json"

if [ -f "$project_dir/my_dataset/metadata.yaml" ]; then
  ros2 run localization_quality offline_evaluator \
    "$project_dir/my_dataset" normal "$work_dir/real_formal.csv" \
    --ros-args --params-file "$project_dir/config/quality_params.yaml" \
    -p "fusion.model_path:=$output_dir/context_mlp_formal.yaml"
  ros2 run localization_quality offline_evaluator \
    "$project_dir/my_dataset" normal "$work_dir/real_rule.csv" \
    --ros-args --params-file "$project_dir/config/quality_params.yaml" \
    -p fusion.mlp_enabled:=false \
    -p fusion.strategy:=rule \
    -p 'fusion.model_path:=""'
  python3 "$project_dir/scripts/summarize_real_bag_smoke.py" \
    --bag-root "$project_dir/my_dataset" \
    --formal-model-csv "$work_dir/real_formal.csv" \
    --rule-csv "$work_dir/real_rule.csv" \
    --expected-frames 336 \
    --output "$output_dir/real_machine_smoke_report.json"
fi

echo "Reproduced archive results: $output_dir"
