# 07-tokenizer

[ZTL-07｜模型｜亲手实现分词](../book/07-tokenizer.md) 正文里的可运行代码，按出现顺序编号。`exNN.out.txt` 是正文里贴出的真实输出，`python3 tools/verify.py 07-tokenizer` 会逐字对比。

| 文件 | 位置 | 内容 |
|---|---|---|
| `ex01.py` | 第 2 章 | 35 行零依赖完整 Byte-level BPE：覆盖训练、全局最小合并秩编码与半截 UTF-8 流式缓冲解码 |
| `ex02.py` | 第 3 章 | 预分词正则围栏与 allowed_special 状态机快轨：验证 16 空格缩进聚合、逐位拆数字与防角色伪造 |
| `ex03.py` | 第 4 章 | 四大前沿架构词表汇率对账与单卡 H100 静态嵌入换动态 KV 盈亏平衡推演脚本 |
