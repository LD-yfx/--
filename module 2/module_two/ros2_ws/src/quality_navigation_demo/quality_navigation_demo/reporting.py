"""Explicit report completion rules, independent of ROS and navigation outcomes."""

STATIC_CASES=('geometric_baseline','quality_aware')
FULL_CASES=STATIC_CASES+('dynamic_top_block','after_obstacle_removal','fully_blocked')
STATIC_CHECKS=('plugin_loaded_and_changes_master_cost','static_goal_reached',
               'quality_changes_route','static_no_collisions')
DYNAMIC_CHECKS=('dynamic_replanned_and_arrived','dynamic_no_collisions',
                'removed_obstacle_cleared','reopened_route_recovers',
                'fully_blocked_stops_with_bounded_failure','fully_blocked_no_collisions')
INSTALLED_CHECK='installed_code_matches_source'

def completion_status(mode, cases, checks, finalized=False, require_installed=False, error=None):
    if mode not in ('full','static_only'):
        raise ValueError('report mode must be full or static_only')
    expected=FULL_CASES if mode=='full' else STATIC_CASES
    required=STATIC_CHECKS+(DYNAMIC_CHECKS if mode=='full' else ())
    if mode=='full' or require_installed:
        required+=(INSTALLED_CHECK,)
    names=tuple(case.get('case') for case in cases)
    missing=[key for key in required if key not in checks]
    completed=bool(finalized and names==expected and not missing and error is None)
    return {'completed':completed,
            'all_passed':completed and all(checks[key] is True for key in required),
            'expected_cases':list(expected),'required_checks':list(required),
            'missing_checks':missing,'finalized':bool(finalized)}
