"""运行全部样例和对照实验：python main.py。"""
import csv
import argparse
import json
import hashlib
import platform
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from algorithms import (astar, TreeNode, tree_search, distance_matrix, game_search,
                        maze_path, terrain_cost, minimax, alpha_beta)
from extra_experiments import run_extra

OUT = Path(__file__).parent / 'results'
ROOT = Path(__file__).parent
MAZE = [list(row) for row in ['S..X.', '.X...', '.X.X.', '...XE', 'XX...']]
TERRAIN = [['S', 1, 1, 8, 1], [1, 8, 3, 1, 1], [1, 8, 1, 8, 1],
           [1, 1, 1, 8, 1], [8, 8, 1, 1, 'E']]


def sample_tree():
    def pair(a, b):
        return TreeNode(children=[TreeNode(a), TreeNode(b)])
    return TreeNode(children=[TreeNode(children=[pair(3, 5), pair(6, 9)]),
                              TreeNode(children=[pair(6, 9), pair(1, 2)])])


def sample_outputs(matrix=None):
    """按题给格式计算四道题的答案；题一保留 Python 坐标元组。"""
    matrix = distance_matrix() if matrix is None else matrix
    return dict(maze=maze_path((0, 0), (3, 4), MAZE),
                terrain=terrain_cost((0, 0), (4, 4), TERRAIN),
                tree=dict(minimax=minimax(sample_tree()), alpha_beta=alpha_beta(sample_tree())),
                distances=matrix,
                treasure={str(depth): {name: game_search(matrix, depth, pruning)[0]
                                      for name, pruning in [('minimax', False), ('alpha_beta', True)]}
                          for depth in (2, 4)})


def print_samples(samples):
    print('题目一：最短路径')
    print(samples['maze'])
    print('题目二：最小移动代价')
    print(samples['terrain'])
    print('题目三：根节点最佳得分')
    for method, value in samples['tree'].items():
        print(f'{method}: {value}')
    print('题目四：6×6 距离矩阵（行是起点，列是终点）')
    matrix = samples['distances']
    labels = list(matrix)
    print('    ' + ' '.join(f'{label:>4}' for label in labels))
    for label in labels:
        print(f'{label:>4}' + ' '.join(f'{matrix[label][end]:>4}' for end in labels))
    for depth, methods in samples['treasure'].items():
        for method, answer in methods.items():
            print(f'H={depth} {method}: {json.dumps(answer, ensure_ascii=False)}')


def timed(function, repetitions=101):
    function()  # 预热，不计时
    times = []
    for _ in range(repetitions):
        start = perf_counter_ns()
        result = function()
        times.append((perf_counter_ns() - start) / 1000)
    return result, round(median(times), 3)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('algorithms.py', 'main.py', 'extra_experiments.py', 'test_algorithms.py')}


def run(validation=None):
    OUT.mkdir(exist_ok=True)
    rows, paths = [], {}
    # 额外场景：较大开放地图、绕墙地图、无解、零代价。
    wall = [[1] * 30 for _ in range(30)]
    for r in range(29):
        wall[r][15] = 'X'
    cases = [('maze', MAZE, (0, 0), (3, 4)), ('terrain', TERRAIN, (0, 0), (4, 4)),
             ('open30', [[1] * 30 for _ in range(30)], (0, 0), (10, 10)),
             ('wall30', wall, (0, 0), (0, 29)),
             ('unreachable', [['S', 'X', 'E']], (0, 0), (0, 2)),
             ('zero_cost', [['S', 0, 0], [1, 1, 'E']], (0, 0), (1, 2))]
    for name, grid, start, end in cases:
        if name == 'maze':
            grid = [[1 if cell in ('.', 'S', 'E') else cell for cell in row] for row in grid]
        for h in (False, True):
            (path, cost, stats), micros = timed(lambda: astar(grid, start, end, h))
            rows.append(dict(task=name, algorithm='astar' if h else 'dijkstra', depth='', order='',
                             value=cost, target='', median_us=micros, **asdict(stats)))
            paths[f'{name}_{h}'] = dict(path=path, cost=cost)
    for order in ('original', 'reverse'):
        for pruning in (False, True):
            (value, stats), micros = timed(lambda: tree_search(sample_tree(), pruning, order))
            rows.append(dict(task='tree', algorithm='alpha_beta' if pruning else 'minimax', depth='',
                             order=order, value=value, target='', median_us=micros, **asdict(stats)))
    # 公平地比较完整 6×6 距离矩阵：两种方法均包含自己的预处理费用。
    matrix_timing = {}
    for reuse in (False, True):
        matrix, micros = timed(lambda: distance_matrix(reuse), repetitions=31)
        matrix_timing['reuse' if reuse else 'repeat'] = micros
    for depth in (2, 4):
        for order in ('original', 'reverse', 'gain', 'bad_gain'):
            for algorithm in ('minimax', 'alpha_beta', 'alpha_beta_cache'):
                (result, stats), micros = timed(lambda: game_search(matrix, depth, algorithm != 'minimax',
                                                                   order, algorithm.endswith('cache')))
                rows.append(dict(task='treasure', algorithm=algorithm, depth=depth, order=order,
                                 value=result['best_value'], target=result['best_target'],
                                 median_us=micros, **asdict(stats)))
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    extra = run_extra(timed)
    samples = sample_outputs(matrix)
    result = dict(paths=paths, distances=matrix, metrics=rows, extra=extra,
                  samples=samples,
                  matrix_timing=dict(median_us=matrix_timing, repetitions=31,
                                     searches=36, includes_preprocessing=True),
                  environment=dict(python=platform.python_version(), system=platform.system()),
                  source_sha256=source_hashes(), validation=validation,
                  timing='101 repetitions after one warmup; median microseconds; includes setup')
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print_samples(samples)
    print('Results saved in', OUT)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-only', action='store_true', help='只输出 PDF 给定样例，不重跑性能实验或覆盖数据')
    args = parser.parse_args()
    if args.samples_only:
        print_samples(sample_outputs())
    else:
        run()
