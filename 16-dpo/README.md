# 16-dpo

[ZTL-16｜后训练｜用偏好对核对 DPO](../book/16-dpo.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 16-dpo` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | PyTorch 2B 拼接分块掩码求和、DPO 四数值手算断言与 Qwen3.8-27B 显存核算 |
| `ex02.py` | 第 4 章 | DPO / RPO / IPO / SimPO 四目标抗冗长过冲核对与 Qwen3.8-27B 8×H100 收益核算 |
