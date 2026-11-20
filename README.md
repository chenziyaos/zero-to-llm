# zero-to-llm

「从零掌握大模型：四笔账学习体系」的 Markdown 版与配套代码。全系列 31 篇：1 篇总览加 30 篇正文，从矩阵乘法一路讲到推理服务、RAG、Agent 和工程交付。每个技术决策都落到效果、显存、时延、成本四笔账上算清楚。

正文在 [`book/`](book/README.md)，从总览 ZTL-00 读起，各篇末尾有上一篇、下一篇的链接。正文里每一段可运行的 Python 代码都抽到了仓库根目录对应章节的目录里，和正文贴出的输出放在一起，可以一键逐字核对。

## 能做什么

- **按顺序读完整课程**：七层由浅入深，L1 数学 → L2 模型结构 → L3 预训练 → L4 后训练 → L5 推理服务 → L6 应用与 Agent → L7 交付。
- **复算正文里的数字**：正文里的表格和结论都来自代码的真实输出，跑一遍就能对上。
- **拿来当速算器**：显存两栏账、算术强度、KV 缓存、排队放大、配对检验这些脚本都只依赖标准库、NumPy 或 PyTorch CPU 版。

## 怎么跑代码

需要 Python 3.10 及以上（输出是在 3.11 上生成的）。

```bash
git clone https://github.com/chenziyaos/zero-to-llm.git && cd zero-to-llm
python3 -m venv .venv && . .venv/bin/activate
pip install numpy==2.4.6
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python3 tools/verify.py              # 全部 81 个脚本，逐字对比正文输出
python3 tools/verify.py 01 07        # 只跑第 01、07 篇
python3 01-matmul/ex01.py            # 单独跑一段
```

`tools/verify.py` 在临时目录里运行每个 `exNN.py`，并把输出与同名的 `exNN.out.txt` 比对。不一致时，它会打印第一处不同的行，退出码为 1。全部跑完在 Apple M 系列 CPU 上约 2 分钟。

## 文件约定

| 路径 | 内容 |
|---|---|
| `book/README.md` | ZTL-00 总览：四笔账、六步闭环、七层地图、度量衡 |
| `book/NN-*.md` | ZTL-01～30 正文 |
| `book/assets/NN-*/fig-KK.png` | 各篇配图，按正文出现顺序编号 |
| `NN-*/exKK.py` | 第 NN 篇正文里第 KK 段可运行代码，首行注释写明篇名和代码标题 |
| `NN-*/exKK.out.txt` | 正文里紧跟这段代码贴出的输出 |
| `NN-*/README.md` | 本篇各段代码在正文哪一章 |
| `tools/verify.py` | 逐段运行并比对输出 |

正文里标成“片段”的代码只为说明写法，不能单独运行，这类代码只留在 `book/` 里，共 2 段。

## 目录

| 目录 | 篇 | 内容 |
|---|---|---|
| `00-overview/` | ZTL-00 总览 | 全系列总纲：四笔账串起从数学基础到工程交付的学习路径 |
| `01-matmul/` | ZTL-01 符号、向量与矩阵乘法 | 张量形状先于公式：点积、矩阵乘、算术强度与低秩分解 |
| `02-softmax-sampling/` | ZTL-02 条件概率、softmax 与采样 | 从 logits 到采样策略与统计误差 |
| `03-entropy-kl/` | ZTL-03 熵、交叉熵、困惑度与 KL | 损失函数在度量什么 |
| `04-backprop/` | ZTL-04 链式法则与 softmax-交叉熵的梯度 | 雅可比链式法则、p−y 梯度与残差 |
| `05-optim-norm/` | ZTL-05 优化、归一化与数值稳定 | AdamW、RMSNorm 与低精度数值稳定 |
| `06-compute-memory-scaling/` | ZTL-06 一张纸算清算力、显存与缩放 | 2P、6PD、显存四件套与缩放定律 |
| `07-tokenizer/` | ZTL-07 亲手实现分词 | Byte-level BPE 原理、实现与词表权衡 |
| `08-attention/` | ZTL-08 把注意力写成矩阵 | 因果掩码、MHA、GQA 与 MLA |
| `09-position/` | ZTL-09 让模型知道顺序 | RoPE 几何原理、KV 缓存下标与长上下文外推 |
| `10-tiny-model/` | ZTL-10 组装小模型并让一个批次下降 | Pre-Norm、SwiGLU 与单批次过拟合验证 |
| `11-pretrain/` | ZTL-11 从语料到一轮小预训练 | 语料清洗、序列打包、学习率调度与监控 |
| `12-multi-gpu/` | ZTL-12 多卡的账和中断后续上 | DP/ZeRO/TP/PP 选型与确定性断点续训 |
| `13-rl-math/` | ZTL-13 奖励、策略梯度与 DPO 的数学 | Bradley-Terry、PPO/GRPO 到 DPO 闭式推导 |
| `14-sft-loss-mask/` | ZTL-14 只对该学的回答算损失 | Chat 模板、标签掩码与多轮打包边界 |
| `15-lora/` | ZTL-15 只训练一小部分参数 | LoRA/QLoRA 的低秩更新、显存收益与合并边界 |
| `16-dpo/` | ZTL-16 用偏好对核对 DPO | 四条序列对数概率、隐式奖励与长度偏置 |
| `17-verifiable-reward/` | ZTL-17 先验证奖励再训练 | 可验证奖励、反例单测集与防刷分 |
| `18-test-time-compute/` | ZTL-18 多想一会儿是否值得 | 思考预算、Pass@k 与边际收益拐点 |
| `19-serving-latency/` | ZTL-19 判断慢在哪，以及请求怎样排队 | Prefill/Decode、分页 KV 与连续批处理 |
| `20-flash-attention/` | ZTL-20 减少注意力的内存搬运 | SRAM 分块、在线 softmax 与 FlashDecoding |
| `21-quantization/` | ZTL-21 用精度换内存要付什么代价 | FP8/INT8/INT4 刻度开销、离群值与 PPL 退化 |
| `22-speculative/` | ZTL-22 让草稿减少解码开销 | 拒绝采样证明、接受率期望与 MTP |
| `23-serving/` | ZTL-23 把模型部署成服务 | SLO 下的 Goodput、前缀缓存与 PD 分离 |
| `24-long-context-moe/` | ZTL-24 长上下文与混合专家怎样改变账单 | 总参管存储、激活参管算力、长窗管 KV |
| `25-retrieval/` | ZTL-25 先把相关材料找出来 | 分块、BM25 加向量混合检索与 Recall@k |
| `26-grounding/` | ZTL-26 让回答有可追溯的依据 | 证据锚点、拒答门禁与幻觉检测 |
| `27-prompt-routing-tools/` | ZTL-27 提示、路由与工具边界 | 结构化输出、大小模型路由与工具安全闸门 |
| `28-eval/` | ZTL-28 先写评测再改系统 | 冻结集、分层指标、配对检验与防污染 |
| `29-agent/` | ZTL-29 Agent 的循环、记忆、权限与成本 | Workflow 与 Agent 分界、上下文压缩与熔断 |
| `30-delivery/` | ZTL-30 交一个可复查的项目 | 证据包、失败样本归档、上线门禁与回滚 |

## 版本基线

- 模型规格以 Hugging Face 上的官方 `config.json` 和技术报告为准，核对于 2026 年 11 月。
- 代码输出的运行环境：macOS arm64，Python 3.11.13，NumPy 2.4.6，torch 2.15.0（CPU）。

## 限制

- **案例读数是假设读数**。后半部分有多篇围绕教学案例展开（编程助手集群、检索库、支付网关、值班 Agent 等）。案例里各版本的准确率、时延、命中率与成本都是示意用的假设读数，不是实测。第 23、25、26、28、29、30 篇在开篇就声明了这一点，第 18～22、27 篇在用到的地方标了“假设读数”。可以复算的只有代码输出与公式推导。
- **换平台可能有末位差异**：浮点结果在其他 CPU、操作系统或 torch 版本上，最后一位可能不同。`verify.py` 会如实报出差异，不做容差放宽。
- **配图是渲染后的 PNG**：原图是 SVG，渲染时用的字体与原文档不同，排版可能略有出入。仓库里没有画图源码。
- **正文是 2026 年 11 月 20 日的快照**：之后原文档的修订不会自动同步过来。
- **没有 Colab notebook，也不需要 GPU**：所有脚本都在 CPU 上跑。

## 常见问题

- **`verify.py` 报某行末位数字不同**：多半是平台或版本差异，见上面的“限制”。先确认 NumPy 和 torch 的版本与 `requirements-cpu.txt` 一致，再看差异是否只在最后一位。
- **只想跑数学篇，不想装 torch**：第 01、02、05、06、07 篇等只依赖标准库或 NumPy，`python3 tools/verify.py 01 02` 只跑指定篇，不会加载 torch。
- **正文和这里的代码对不上**：以 `book/` 里的正文为准，代码和输出都是从正文原样抽出的；发现不一致请提 issue。

## 许可

MIT，见 [LICENSE](LICENSE)。
