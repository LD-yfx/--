# A complete sparse snapshot, not an incremental update.
# Cell index = y * info.width + x. Indices are unique and ascending.
# All arrays have the same length. Omitted cells are unknown, even if an older
# snapshot contained them. A snapshot with no indices clears all known cells.
# Quality is a score in [0, 1], higher is better, not occupancy probability.
std_msgs/Header header
nav_msgs/MapMetaData info

# "cumulative": all accepted observations since this map was reset.
# "windowed": retained observations from the configured rolling time window,
# also bounded by the publisher's per-cell sample capacity (latest retained).
# Variance is unbiased sample variance (zero for a single observation).
# Neither mode implies independent samples or a statistical confidence interval.
string statistics_mode
uint32[] indices
float32[] mean
float32[] variance
uint64[] sample_count

# Per-cell time of its last actual observation; never map publication time.
# Same clock domain as header.stamp (ROS simulation time if use_sim_time).
builtin_interfaces/Time[] last_observed
