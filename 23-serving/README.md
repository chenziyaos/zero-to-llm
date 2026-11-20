# 23-serving

[ZTL-23｜系统｜把模型部署成服务](../book/23-serving.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 23-serving` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 仿真 24 QPS 突发洪峰下无界排队傻跑 vs 队列上限快返 429 配断连即回收的 Goodput 差距 |
| `ex02.py` | 第 4 章 | 四代架构 16K 跨机 RDMA 传输耗时、xPyD 配比及含潮汐因子百万词元真实成本核算器 |
