# TF publisher audit for ROS2 Humble

This read-only benchmark node subscribes to `/tf` and transient-local `/tf_static`.
It uses `rclcpp::MessageInfo` to associate every observed parent/child frame pair
with the actual DDS publisher GID. The installed Humble `rclpy` callback API does
not expose that GID, so topic-wide publisher lists alone cannot prove frame ownership.

`/benchmark/tf_authorities` is reliable, transient-local `std_msgs/String` JSON,
published every 0.5 wall seconds. `stamp` uses the configured ROS clock.
`tf_authorities` and `publishers` retain all observed owners and their identities
throughout one node lifetime, including publishers that have exited. GIDs are the
complete `RMW_GID_STORAGE_SIZE` bytes in lowercase hexadecimal (24 bytes here),
matching Python topic endpoint GIDs. `runtime_executable_path` resolves `/proc/self/exe`.

The benchmark runner must verify the audit topic's publisher, freshness, and the
expected frame owners. An unknown GID remains visible and must not be inferred to
be a trusted node. This node neither publishes transforms nor subscribes to truth.
Start a new node for each episode; process restart resets its cumulative record.

The actual DDS regression is `module_two/tests/test_tf_authority_audit.py`. It
checks endpoint-GID correspondence, late-join static TF delivery, and retention
of a duplicate frame owner after that publisher disappears.
