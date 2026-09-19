"""Independent graph oracle checks objective optimality and corner constraints."""
import heapq
import math
import numpy as np
import pytest
from quality_aware_navigation.planning import astar, inflate_obstacles, path_is_valid, Replanner


def dijkstra_oracle(mask, cost, start, goal, resolution, weight):
    # Build a graph separately, then run unheuristic shortest-path search.
    vertices = [(x, y) for y, row in enumerate(mask) for x, blocked in enumerate(row) if not blocked]
    if start not in vertices or goal not in vertices:
        return math.inf
    edges = {v: [] for v in vertices}
    for a in vertices:
        for b in vertices:
            dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
            if max(dx, dy) != 1:
                continue
            if dx and dy and (mask[a[1], b[0]] or mask[b[1], a[0]]):
                continue
            mean_soft_cost = (int(cost[a[1], a[0]]) + int(cost[b[1], b[0]])) / 2 / 252
            edges[a].append((b, resolution * math.dist(a, b) * (1 + weight * mean_soft_cost)))
    best = {start: 0.}
    queue = [(0., start)]
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != best[vertex]:
            continue
        if vertex == goal:
            return distance
        for neighbour, length in edges[vertex]:
            candidate = distance + length
            if candidate < best.get(neighbour, math.inf):
                best[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return math.inf


@pytest.mark.parametrize('seed', range(20))
def test_astar_matches_independent_dijkstra(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((6, 7)) < .27
    mask[0, 0] = mask[-1, -1] = False
    costs = rng.integers(0, 253, mask.shape, dtype=np.uint8)
    weight, resolution = (seed % 5), .25
    actual = astar(mask, costs, (0, 0), (6, 5), resolution, weight)
    expected = dijkstra_oracle(mask, costs, (0, 0), (6, 5), resolution, weight)
    assert actual.objective == pytest.approx(expected)
    assert actual.success == math.isfinite(expected)
    if actual.success:
        assert path_is_valid(actual.path, mask)
        assert actual.path[0] == (0, 0) and actual.path[-1] == (6, 5)


def test_diagonal_does_not_cut_obstacle_corner():
    blocked = np.array([[False, True], [True, False]])
    assert not astar(blocked, np.zeros((2, 2)), (0, 0), (1, 1)).success


def test_soft_cost_never_opens_a_hard_obstacle():
    blocked = np.zeros((5, 7), dtype=bool)
    blocked[:, 3] = True
    assert not astar(blocked, np.zeros(blocked.shape), (1, 2), (5, 2)).success


def test_inflation_and_boundary_clearance():
    blocked = np.zeros((11, 11), dtype=bool)
    blocked[5, 5] = True
    small = inflate_obstacles(blocked, 1, .4)
    large = inflate_obstacles(blocked, 1, 1.1)
    assert small[5, 4] and small[4, 5]
    assert np.all(large[small])
    assert np.all(large[0]) and np.all(large[:, 0])
    assert np.array_equal(inflate_obstacles(blocked, 1, 0), blocked)


def test_dynamic_block_forces_immediate_replan_even_inside_hold_interval():
    blocked = np.zeros((7, 11), dtype=bool)
    cost = np.zeros(blocked.shape)
    replanner = Replanner(min_interval=100)
    first, _ = replanner.update(blocked, cost, (1, 3), (9, 3), 0)
    blocked[3, 5] = True
    second, reason = replanner.update(blocked, cost, (1, 3), (9, 3), .1)
    assert reason == 'blocked_or_initial'
    assert second != first and path_is_valid(second, blocked)
    blocked[:, 5] = True
    stopped, reason = replanner.update(blocked, cost, (1, 3), (9, 3), .2)
    assert stopped == [] and reason == 'stop_no_path'
    blocked[:, 5] = False
    recovered, _ = replanner.update(blocked, cost, (1, 3), (9, 3), .3)
    assert recovered and path_is_valid(recovered, blocked)


def test_clearance_and_quality_improvements_respect_interval_then_switch():
    blocked = np.zeros((9, 15), dtype=bool)
    cost = np.zeros(blocked.shape)
    replanner = Replanner(min_interval=1., improvement_fraction=.01)
    original, _ = replanner.update(blocked, cost, (1, 4), (13, 4), 0)
    cost[4, 3:12] = 252
    held, reason = replanner.update(blocked, cost, (1, 4), (13, 4), .5)
    assert held == original and reason == 'held_interval'
    switched, reason = replanner.update(blocked, cost, (1, 4), (13, 4), 2)
    assert switched != original and reason == 'quality_improvement'
    cost[:] = 0
    recovered, reason = replanner.update(blocked, cost, (1, 4), (13, 4), 4)
    assert recovered == original and reason == 'quality_improvement'


def test_clock_reset_does_not_hold_stale_route():
    blocked = np.zeros((5, 8), dtype=bool)
    cost = np.zeros(blocked.shape)
    replanner = Replanner(min_interval=100)
    replanner.update(blocked, cost, (1, 2), (6, 2), 500)
    path, reason = replanner.update(blocked, cost, (1, 2), (6, 2), 1)
    assert path and reason == 'blocked_or_initial'
