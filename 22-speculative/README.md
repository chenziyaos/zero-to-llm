# 22-speculative

[ZTL-22｜系统｜让草稿减少解码开销](../book/22-speculative.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 22-speculative` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 修正拒绝采样 10 万次蒙特卡洛随机模拟核验 |
| `ex02.py` | 第 3 章 | 串行步长寻优与 MTP 对 块扩散草稿 的 TPOT 对账 |
| `ex03.py` | 第 4 章 | ReplaySSM 逐比特精确回滚与动态批度熔断 |
