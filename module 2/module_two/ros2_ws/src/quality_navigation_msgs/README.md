# Quality navigation messages

`quality_navigation_msgs/msg/QualityGrid` carries a **complete sparse snapshot**
of localization quality statistics. It is not an obstacle map and is not a delta.

- `indices[i] = y * info.width + x`; indices are unique and ascending.
- Every value array has exactly `indices.size()` elements.
- Omitted cells are unknown. The receiver replaces its previous snapshot.
- `mean` is a quality score in `[0,1]`; higher means better quality.
- `variance` is the unbiased sample variance, not pose covariance. It is zero
  when `sample_count` is one; this does not imply high certainty.
- `last_observed` records each cell's last actual sensor observation. All times
  share the clock domain of `header.stamp`; use `/clock` in simulation.
- `statistics_mode` is `windowed` (the default publisher mode) or `cumulative`.
  Windowed statistics retain recent samples within a configured duration and
  per-cell capacity; cumulative statistics include all samples since reset.
  Counts do not establish independent samples or calibrated confidence.
- Window expiration forgets both favorable and unfavorable evidence. A cell
  whose samples all expire becomes unknown; its downstream cost may increase
  or decrease to the configured unknown cost. Age monotonicity of a cost model
  with fixed statistics does not imply monotonicity across window updates.
- `header.stamp` is the sensor evaluation time associated with the snapshot,
  not wall-clock publication time. It must not replace per-cell timestamps.

The module-one publisher uses reliable, transient-local, keep-last-one QoS.
Consumers should validate geometry, array lengths, finite values, unique valid
indices, positive sample counts and timestamps before replacing a valid map.
Reject unsupported statistics modes rather than silently changing semantics.
