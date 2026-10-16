"""实验二：A*、Minimax、Alpha-beta。仅依赖 Python 标准库。"""
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from numbers import Integral
from types import MappingProxyType


@dataclass
class Stats:
    nodes: int = 0
    generated: int = 0
    leaves: int = 0
    cutoffs: int = 0
    cache_hits: int = 0


@dataclass(frozen=True)
class PreparedGrid:
    """一次转换后的地图快照，可供多次寻路复用；权重不能被搜索修改。"""
    weights: object
    minimum: int
    minimum_count: int
    next_minimum: int


def prepare_grid(grid):
    if isinstance(grid, PreparedGrid):
        return grid
    cols = len(grid[0]) if grid else 0
    if any(len(row) != cols for row in grid):
        raise ValueError('地图必须为矩形')

    def weight(value):
        if value == 'X':
            return None
        if value == 'E':
            return 0
        if isinstance(value, str) and (value in ('.', 'S', 'A', 'B') or
                                       (value.startswith('T') and value[1:].isdigit())):
            return 1
        if isinstance(value, bool) or not isinstance(value, (str, Integral)):
            raise ValueError('进入代价必须为非负整数')
        try:
            number = int(value)
        except ValueError as error:
            raise ValueError(f'无法识别的地图格子：{value!r}') from error
        if number < 0:
            raise ValueError('A* 不支持负进入代价')
        return number

    weights = {(r, c): weight(value) for r, row in enumerate(grid) for c, value in enumerate(row)}
    counts = {}
    for value in weights.values():
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    smallest = sorted(counts)[:2]
    minimum = smallest[0] if smallest else 0
    return PreparedGrid(MappingProxyType(weights), minimum, counts.get(minimum, 0),
                        smallest[1] if len(smallest) > 1 else 0)


def astar(grid, start, end, heuristic=True, tie_break='fifo'):
    """接收原地图或 prepare_grid 的结果。X 不通，E 为 0，其余特殊格为 1。

    返回 (路径, 总代价, 统计)。nodes 为有效出队数，包含终点。
    使用惰性删除和重新入队，不依赖启发函数一致性关闭节点。
    """
    stats = Stats()
    if tie_break not in ('fifo', 'larger_g'):
        raise ValueError('tie_break 必须为 fifo 或 larger_g')
    prepared = prepare_grid(grid)
    weights = prepared.weights
    start, end = tuple(start), tuple(end)
    if any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in p)
           for p in (start, end)):
        raise ValueError('起终点必须为两个整数组成的坐标')
    if start not in weights or end not in weights or weights[start] is None or weights[end] is None:
        return [], -1, stats
    # 最后一步独立计价；其余步使用除终点外最小进入代价。
    minimum = (prepared.next_minimum if weights[end] == prepared.minimum and prepared.minimum_count == 1
               else prepared.minimum)

    def h(p):
        distance = abs(p[0] - end[0]) + abs(p[1] - end[1])
        return ((distance - 1) * minimum + weights[end]) if heuristic and distance else 0

    serial = count()
    # 只调整 f 相同时的顺序，不改变 f=g+h，也不放大启发函数。
    queue = [(h(start), 0, next(serial), 0, start)]
    best, parent = {start: 0}, {}
    stats.generated = 1
    while queue:
        _, _, _, cost, current = heappop(queue)
        if cost != best[current]:
            continue
        stats.nodes += 1
        if current == end:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1], cost, stats
        r, c = current
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            step = weights.get(neighbor)
            if step is None:
                continue
            candidate = cost + step
            if candidate < best.get(neighbor, inf):
                best[neighbor], parent[neighbor] = candidate, current
                secondary = -candidate if tie_break == 'larger_g' else 0
                heappush(queue, (candidate + h(neighbor), secondary, next(serial), candidate, neighbor))
                stats.generated += 1
    return [], -1, stats


def maze_path(start, end, grid):
    """题目一：含起终点的最短路径，无解返回 []。"""
    unit_grid = [[1 if cell in ('.', 'S', 'E') else cell for cell in row] for row in grid]
    return astar(unit_grid, start, end)[0]


def terrain_cost(start, end, grid):
    """题目二：最小进入代价和，不计算起点，无解返回 -1。"""
    return astar(grid, start, end)[1]


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = list(children) if children is not None else []


def tree_search(root, pruning=True, order='original'):
    """题目三，默认严格从左到右；reverse 仅用于对照实验。"""
    if order not in ('original', 'reverse'):
        raise ValueError('树的访问顺序必须为 original 或 reverse')
    stats = Stats()

    def visit(node, maximizing, alpha, beta):
        stats.nodes += 1
        if not node.children:
            if node.value is None:
                raise ValueError('叶节点必须提供 value')
            stats.leaves += 1
            return node.value
        value = -inf if maximizing else inf
        children = node.children if order == 'original' else reversed(node.children)
        for i, child in enumerate(children):
            result = visit(child, not maximizing, alpha, beta)
            value = max(value, result) if maximizing else min(value, result)
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if pruning and alpha >= beta and i + 1 < len(node.children):
                stats.cutoffs += 1
                break
        return value

    return visit(root, True, -inf, inf), stats


def alpha_beta(root):
    return tree_search(root)[0]


def minimax(root):
    return tree_search(root, pruning=False)[0]


POINTS = {'A': (0, 0), 'B': (6, 6), 'T1': (0, 6), 'T2': (2, 2), 'T3': (4, 4), 'T4': (6, 0)}
VALUES = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}
BOARD = [
    ['A', 1, 1, 'X', 2, 1, 'T1'],
    [1, 'X', 6, 'X', 1, 'X', 1],
    [1, 1, 'T2', 1, 1, 1, 1],
    ['X', 1, 'X', 8, 'X', 1, 'X'],
    [1, 1, 1, 1, 'T3', 1, 1],
    [1, 'X', 9, 'X', 1, 'X', 1],
    ['T4', 1, 1, 'X', 1, 1, 'B'],
]


def distance_matrix(reuse_preprocessing=True):
    grid = prepare_grid(BOARD) if reuse_preprocessing else BOARD
    return {a: {b: astar(grid, p, q)[1] for b, q in POINTS.items()} for a, p in POINTS.items()}


@dataclass(frozen=True)
class GameState:
    a: str = 'A'
    b: str = 'B'
    remaining: tuple = tuple(VALUES)
    score_a: int = 0
    score_b: int = 0
    player: str = 'A'


def game_search(distances, depth=4, pruning=True, order='original', cache=False):
    """精确搜索。缓存只存已完全搜索的节点，避免把剪枝边界误当精确值。

    缓存值为未来收益差，故键可省略已累计收益；返回值包含当前收益差。
    """
    if depth < 0:
        raise ValueError('深度不能为负')
    if order not in ('original', 'reverse', 'gain', 'bad_gain'):
        raise ValueError('无法识别的宝藏访问顺序')
    stats, table = Stats(), {}

    def visit(state, remaining_depth, alpha, beta):
        stats.nodes += 1
        current = state.score_a - state.score_b
        key = (state.a, state.b, state.remaining, state.player, remaining_depth)
        if cache and key in table:
            stats.cache_hits += 1
            delta, target = table[key]
            return current + delta, target, True
        if remaining_depth == 0 or not state.remaining:
            stats.leaves += 1
            if cache:
                table[key] = (0, None)
            return current, None, True
        maximizing = state.player == 'A'
        position = state.a if maximizing else state.b
        targets = [t for t in state.remaining if distances[position][t] >= 0]
        if order == 'reverse':
            targets.reverse()
        elif order in ('gain', 'bad_gain'):
            targets.sort(key=lambda t: VALUES[t] - distances[position][t], reverse=order == 'gain')
        value, target, exact = (-inf if maximizing else inf), None, True
        for i, t in enumerate(targets):
            gain = VALUES[t] - distances[position][t]
            child = GameState(t if maximizing else state.a, state.b if maximizing else t,
                              tuple(x for x in state.remaining if x != t),
                              state.score_a + (gain if maximizing else 0),
                              state.score_b + (0 if maximizing else gain), 'B' if maximizing else 'A')
            score, _, child_exact = visit(child, remaining_depth - 1, alpha, beta)
            exact = exact and child_exact
            # 并列保留访问顺序中的首个；剪枝返回的界不能用于并列替换。
            if target is None or (score > value if maximizing else score < value):
                value, target = score, t
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if pruning and alpha >= beta and i + 1 < len(targets):
                stats.cutoffs += 1
                exact = False
                break
        if not targets:
            stats.leaves += 1
            value = current
        if cache and exact:
            table[key] = (value - current, target)
        return value, target, exact

    value, target, _ = visit(GameState(), depth, -inf, inf)
    return {'best_target': target, 'best_value': value}, stats
