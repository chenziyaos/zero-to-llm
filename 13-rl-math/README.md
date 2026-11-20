# 13-rl-math

[ZTL-13｜后训练｜奖励、策略梯度与 DPO 的数学](../book/13-rl-math.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 13-rl-math` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 3 章 | PyTorch 完整可运行实现：GRPO 组内优势归一化、双侧裁剪、Schulman k3 散度与 Miles TIS 训推修正 |
| `ex02.py` | 第 4 章 | 玻尔兹曼闭式最优分布、配分函数 Z(x) 精确对消与 DPO 错判样本动态梯度权重数值验证 |
| `ex03.py` | 第 5 章 | 单机 8×H100 训练稠密混合 27B：四模型 PPO、离线 DPO 与共置 SGLang 睡眠唤醒 GRPO 对账脚本 |
