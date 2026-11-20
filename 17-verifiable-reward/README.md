# 17-verifiable-reward

[ZTL-17｜后训练｜先验证奖励再训练](../book/17-verifiable-reward.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 17-verifiable-reward` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 3 章 | 45 行纯标准库平衡花括号提取器、一次性令牌隔离代码沙箱与 T1～T8 红队反例全绿单测 |
| `ex02.py` | 第 4 章 | 融合 DAPO 零方差死题过滤、Clip-Higher(0.20/0.28) 非对称裁剪、词元级归约与 k3 KL 的 GRPO 更新步 |
| `ex03.py` | 第 5 章 | 32×H100 (8×TP4) 训练案例模型 对比 Qwen3-32B 带宽下限及 V0/V1/V2 事故修复四笔账核算器 |
