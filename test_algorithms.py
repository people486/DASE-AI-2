"""独立最优性基准：BFS / Bellman-Ford / 穷举博弈。"""
import random
import unittest
from collections import deque
from itertools import permutations
from math import inf
from algorithms import (astar, prepare_grid, maze_path, terrain_cost, TreeNode,
                        alpha_beta, minimax, tree_search, distance_matrix, game_search, VALUES)
from main import MAZE, TERRAIN, sample_tree, sample_outputs


class SearchTests(unittest.TestCase):
    def check_path(self, grid, start, end, path, cost):
        """直接按原输入检查路径，避免只比较算法返回的代价。"""
        if cost == -1:
            self.assertEqual(path, [])
            return
        self.assertEqual((path[0], path[-1]), (start, end))
        actual = 0
        for i, (r, c) in enumerate(path):
            self.assertTrue(0 <= r < len(grid) and 0 <= c < len(grid[0]))
            cell = grid[r][c]
            self.assertNotEqual(cell, 'X')
            if i:
                previous = path[i - 1]
                self.assertEqual(abs(r - previous[0]) + abs(c - previous[1]), 1)
                actual += 0 if cell == 'E' else 1 if cell in ('.', 'S', 'A', 'B') or str(cell).startswith('T') else int(cell)
        self.assertEqual(actual, cost)

    def test_degenerate_paths(self):
        cases = [([[1]], (0, 0), (0, 0), 0), ([['X']], (0, 0), (0, 0), -1),
                 ([['S', 3, 'E']], (0, 0), (0, 2), 3),
                 ([['S'], [0], ['E']], (0, 0), (2, 0), 0),
                 ([['S', 'X']], (0, 0), (0, 1), -1),
                 ([['X', 'E']], (0, 0), (0, 1), -1),
                 ([[1, 1]], (-1, 0), (0, 1), -1)]
        for grid, start, end, expected in cases:
            for source in (grid, prepare_grid(grid)):
                for tie in ('fifo', 'larger_g'):
                    path, cost, _ = astar(source, start, end, tie_break=tie)
                    self.assertEqual(cost, expected)
                    self.check_path(grid, start, end, path, cost)

    def test_invalid_inputs(self):
        for grid in ([[1], [1, 1]], [[], [1]], [[-1]], [[1.5]], [[True]], [['?']]):
            with self.assertRaises(ValueError):
                prepare_grid(grid)
        with self.assertRaises(ValueError):
            astar([[1]], (0.5, 0), (0, 0))
        with self.assertRaises(ValueError):
            astar([[1]], (0, 0), (0, 0), tie_break='unknown')
        with self.assertRaises(ValueError):
            tree_search(sample_tree(), order='unknown')
        with self.assertRaises(ValueError):
            game_search({}, order='unknown')

    def test_prepared_grid_reuse(self):
        grid = [[1, 1], [1, 1]]
        prepared = prepare_grid(grid)
        grid[0][1] = grid[1][0] = 99
        self.assertEqual(astar(prepared, (0, 0), (1, 1))[1], 2)
        self.assertEqual(astar(grid, (0, 0), (1, 1))[1], 100)
        self.assertEqual(distance_matrix(True), distance_matrix(False))

    def test_samples_and_boundaries(self):
        samples = sample_outputs()
        self.assertTrue(all(isinstance(p, tuple) for p in samples['maze']))
        self.assertEqual((len(samples['maze']) - 1, samples['terrain']), (7, 7))
        self.assertEqual(samples['tree'], {'minimax': 5, 'alpha_beta': 5})
        self.assertEqual(len(samples['distances']), 6)
        self.assertTrue(all(len(row) == 6 for row in samples['distances'].values()))
        for depth, target, value in [('2', 'T2', -1), ('4', 'T3', 3)]:
            for answer in samples['treasure'][depth].values():
                self.assertEqual(answer, {'best_target': target, 'best_value': value})
        path = maze_path((0, 0), (3, 4), MAZE)
        self.assertEqual(len(path) - 1, 7)
        self.assertEqual(terrain_cost((0, 0), (4, 4), TERRAIN), 7)
        self.assertEqual(alpha_beta(sample_tree()), 5)
        self.assertEqual(maze_path((0, 0), (0, 2), [['S', 'X', 'E']]), [])
        self.assertEqual(terrain_cost((0, 0), (0, 2), [['S', 'X', 'E']]), -1)
        self.assertEqual(astar([[1]], (0, 0), (0, 0))[:2], ([(0, 0)], 0))
        self.assertEqual(astar([], (0, 0), (0, 0))[1], -1)
        with self.assertRaises(ValueError):
            astar([[1, -1]], (0, 0), (0, 1))

    def test_random_mazes_against_bfs(self):
        rng = random.Random(42)
        for _ in range(150):
            grid = [['X' if rng.random() < .3 else '.' for _ in range(7)] for _ in range(7)]
            grid[0][0] = grid[6][6] = '.'
            queue, distances = deque([(0, 0)]), {(0, 0): 0}
            while queue:
                r, c = queue.popleft()
                for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if 0 <= p[0] < 7 and 0 <= p[1] < 7 and grid[p[0]][p[1]] != 'X' and p not in distances:
                        distances[p] = distances[(r,c)] + 1
                        queue.append(p)
            for source in (grid, prepare_grid(grid)):
                for tie in ('fifo', 'larger_g'):
                    path, cost, _ = astar(source, (0,0), (6,6), tie_break=tie)
                    self.assertEqual(cost, distances.get((6,6), -1))
                    self.check_path(grid, (0,0), (6,6), path, cost)

    def test_random_weights_against_bellman_ford(self):
        rng = random.Random(2026)
        for _ in range(100):
            grid = [[rng.choice([0, 1, 3, 8, 'X']) for _ in range(5)] for _ in range(5)]
            grid[0][0], grid[4][4] = 1, 0
            d = {(0,0): 0}
            for _ in range(24):
                previous = d.copy()
                for (r,c), cost in previous.items():
                    for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                        if 0 <= p[0] < 5 and 0 <= p[1] < 5 and grid[p[0]][p[1]] != 'X':
                            d[p] = min(d.get(p, inf), cost + grid[p[0]][p[1]])
            for source in (grid, prepare_grid(grid)):
                for tie in ('fifo', 'larger_g'):
                    path, cost, _ = astar(source, (0,0), (4,4), tie_break=tie)
                    self.assertEqual(cost, d.get((4,4), -1))
                    self.check_path(grid, (0,0), (4,4), path, cost)

    def test_random_trees(self):
        rng = random.Random(99)
        def build(depth):
            return TreeNode(rng.randint(-20, 20)) if not depth else TreeNode(children=[build(depth-1) for _ in range(rng.randint(1,4))])
        for _ in range(100):
            tree = build(4)
            expected = minimax(tree)
            for order in ('original','reverse'):
                self.assertEqual(tree_search(tree, True, order)[0], expected)

    def test_game_exhaustive(self):
        matrix = distance_matrix()
        for depth, expected in [(2, ('T2', -1)), (4, ('T3', 3))]:
            # 对每个合法行动序列独立累计收益，再由后向归纳求根值。
            scores = {}
            for sequence in permutations(VALUES, depth):
                a, b, score = 'A', 'B', 0
                for turn, target in enumerate(sequence):
                    if turn % 2 == 0:
                        score += VALUES[target] - matrix[a][target]
                        a = target
                    else:
                        score -= VALUES[target] - matrix[b][target]
                        b = target
                scores[sequence] = score
            for level in range(depth-1, -1, -1):
                reduced = {}
                for sequence, score in scores.items():
                    prefix = sequence[:level]
                    op = max if level % 2 == 0 else min
                    reduced[prefix] = op(reduced.get(prefix, -inf if level % 2 == 0 else inf), score)
                scores = reduced
            self.assertEqual(scores[()], expected[1])
            for order in ('original','reverse','gain','bad_gain'):
                for pruning in (False, True):
                    for cache in (False, True):
                        result, _ = game_search(matrix, depth, pruning, order, cache)
                        self.assertEqual((result['best_target'], result['best_value']), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
