# 14-sft-loss-mask

[ZTL-14｜后训练｜只对该学的回答算损失](../book/14-sft-loss-mask.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 14-sft-loss-mask` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 构造多轮工具排障轨迹无泄漏 labels 并断言背景与工具观察梯度严格归零 |
| `ex02.py` | 第 4 章 | PyTorch 验证微批 mean 千倍权重畸变 vs 全局 ∑m_t 归约零误差等价及吞吐核算 |
