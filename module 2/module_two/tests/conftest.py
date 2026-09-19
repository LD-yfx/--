"""Run algorithm tests without a sourced ROS installation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'quality_aware_navigation'))
sys.path.insert(0, str(ROOT / 'experiments'))
