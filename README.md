# 实验二 搜索算法

实现 A* 迷宫寻路、加权地形、Minimax、Alpha-beta 与宝藏分配。

## 运行

Python 3.10+，仅依赖标准库，无需安装第三方库。在本文件夹运行：

```sh
python main.py --samples-only
python -m unittest -v
```

第一条输出样例，不覆盖已保存结果；第二条执行正确性测试。

重新运行全部性能实验并更新结果：

```sh
python main.py
```

## 文件说明

- algorithms.py：搜索算法与博弈实现。
- main.py：样例、性能对照和结果保存。
- extra_experiments.py：不同地图和平局处理策略的补充实验。
- test_algorithms.py：样例、边界和独立算法对照测试。
- results/results.json：完整结果、运行环境、源码摘要及测试记录。
- results/metrics.csv：基础实验指标。
- results/extra_metrics.csv：补充实验指标。

## 样例结果与规则

题一最短路径 7 步，题二费用 7，题三根值 5。题四 H=2 选择 T2、收益差 -1；H=4 选择 T3、收益差 3，两种博弈算法一致。

题一每步费用为 1，题二进入 E 为 0，题四特殊格进入费用为 1。起点不计费；经过其他宝藏不自动收集，已获取位置仍可通行。

基础配置预热后运行 101 次，补充实验每图运行 31 次，耗时取中位数，随机种子固定为 0 至 9。耗时随机器和负载变化。
