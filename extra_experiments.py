"""只增加一个改动：f 相同时优先较大 g；在相同地图上配对比较。"""
import csv
import random
from pathlib import Path
from statistics import mean, median
from algorithms import astar


def make_cases():
    yield '开放80', 0, [[1] * 80 for _ in range(80)], (0, 0), (79, 79)
    wall = [[1] * 80 for _ in range(80)]
    for r in range(79):
        wall[r][40] = 'X'
    yield '绕墙80', 0, wall, (0, 0), (0, 79)
    for size in (40, 80):
        for seed in range(10):
            rng = random.Random(seed)
            grid = [['X' if rng.random() < .22 else rng.choice([1, 1, 1, 3, 8])
                     for _ in range(size)] for _ in range(size)]
            # 固定一条可行路线，避免用丢弃无解地图的方式筛选样本。
            for k in range(size):
                grid[0][k] = grid[k][size - 1] = 1
            yield f'随机{size}', seed, grid, (0, 0), (size - 1, size - 1)


def run_extra(timed):
    records = []
    configs = [('Dijkstra', False, 'fifo'), ('A* FIFO', True, 'fifo'),
               ('A* 较大g', True, 'larger_g')]
    for scene, seed, grid, start, end in make_cases():
        expected = None
        for method, heuristic, tie in configs:
            (path, cost, stats), micros = timed(
                lambda: astar(grid, start, end, heuristic, tie), repetitions=31)
            if expected is None:
                expected = cost
            if cost != expected or not path:
                raise AssertionError(f'{scene}/{seed}/{method} 未得到相同的可达最优代价')
            records.append(dict(scene=scene, seed=seed, method=method, cost=cost,
                                nodes=stats.nodes, median_us=micros))
    summary = []
    for scene in dict.fromkeys(r['scene'] for r in records):
        group = [r for r in records if r['scene'] == scene]
        item = dict(scene=scene, maps=len(group) // 3,
                    mean_cost=round(mean(r['cost'] for r in group if r['method'] == 'Dijkstra'), 1))
        for method, _, _ in configs:
            samples = [r for r in group if r['method'] == method]
            item[method] = dict(mean_nodes=round(mean(r['nodes'] for r in samples), 1),
                                median_us=round(median(r['median_us'] for r in samples), 1))
        summary.append(item)
    out = Path(__file__).parent / 'results'
    out.mkdir(exist_ok=True)
    with (out / 'extra_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return dict(summary=summary, records=records, seeds=list(range(10)), repetitions=31,
                note='Same maps for all methods; random maps have 22% obstacles, weights 1/3/8; '
                     'top row and right column set to 1 for reachability; '
                     'mean nodes and median of per-map median microseconds; includes grid setup')
