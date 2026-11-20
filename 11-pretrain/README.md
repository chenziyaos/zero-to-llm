# 11-pretrain

[ZTL-11｜训练｜从语料到一轮小预训练](../book/11-pretrain.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 11-pretrain` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 对齐 Qwen3.8-27B（248,320 词表）的 uint32 memmap 打包、cu_seqlens 与线性状态 reset_state 双隔离验证 |
| `ex02.py` | 第 4 章 | 含全局有效词数无偏归约、BF16 混合精度、裁剪前范数捕获、WSD 与真 MFU 打点的标准主循环 |
| `ex03.py` | 第 5 章 | 一轮 0.5B×10B 预训练四档配置真 MFU、8×H100 墙钟耗时与成本自动复算核验脚本 |
