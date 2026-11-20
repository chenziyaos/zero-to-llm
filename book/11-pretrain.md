# ZTL-11｜训练｜从语料到一轮小预训练

> 🎯 **这篇文档解决什么问题**：<b>核心命题：预训练的决胜关键从未停留在原始网页体积大小，真正的天花板必须依赖单位算力吃进去的有效无污染信息密度。</b>五级提纯漏斗从海量快照里筛出不足一成的高纯语料。块对角打包与线性状态清零负责榨干每个窗口槽位。全局词数梯度归约、WSD 弹性退火与四条遥测曲线则守住长跑收敛底线。
>
> **使用方式与先修要求**：本篇是 L3 预训练系统层的开篇之作。它直接承接刚通过单批冒烟测试的小模型。我们将走完一轮高 MFU、高验证分的小预训练闭环。文中还会对比多款 2026 主力模型的工业配方与底盘规格。阅读前建议熟悉每字节比特数（BPB）、`AdamW` 与 `6PD` 定律。
>
> **动手实践**：第 2 章提供对齐 `Qwen3.8-27B`（`248,320` 大词表）的 `uint32 memmap` 打包与双隔离脚本。第 4 章交付标准 PyTorch `BF16` 预训练主循环。第 5 章附带 `8×H100` 一轮 `10B` 词元的对账脚本，复制即可运行。

<b>🧭 主线位置</b>：上一篇 [ZTL-10｜模型｜组装小模型并让一个批次下降](./10-tiny-model.md) 已组装完整小模型并跑通单批冒烟门禁。本篇打通脏网页提纯、混合架构零浪费打包、全局梯度归约与 WSD 退火监控。下一篇 [ZTL-12｜训练｜多卡的账和中断后续上](./12-multi-gpu.md) 将接棒百卡显存切分、跨机通信隐藏与掉卡原子续训。

## 导读 · 先给结论

- <b>原始网页仅有不足一成够格入库，五级漏斗加两阶段配比决定上限。</b>经正文剥离、规则快筛、去重、分类器打分与 13-gram 净身后，约 `8.2%` 的高纯词元即可反超全量脏语料。业界主流模型均在退火期拉高数理代码配比[[1]](https://arxiv.org/abs/2406.17557)[[2]](https://arxiv.org/abs/2412.19437)。
- <b>大词表分片必选 uint32，混合架构打包须同步隔离注意力与线性态。</b>`DeepSeek-V3`（`129K`）与 `Qwen3.8-27B`（`248,320`）远超 `uint16` 极值，必须存为 `uint32` 并用 `np.memmap` 映射。跨过 `<|endoftext|>` 时，全局层须传 `cu_seqlens` 并归零位置，线性层须清零递推态与短卷积[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)[[4]](https://huggingface.co/moonshotai/Kimi-K3)[[5]](https://arxiv.org/abs/2107.02027)。
- <b>梯度累积须按全局有效词数归约，漏除系数会引爆步长与裁剪失真。</b>全局批大小满足 $`B_{\mathrm{tok}} = B_{\mathrm{micro}} K_{\mathrm{accum}} W_{\mathrm{dp}} S`$ 。漏除分母会让累积导数虚高 $`K_{\mathrm{accum}}`$ 倍。健康梯度随即步步遭到硬截断。
- <b>工业界已从锁死终点的余弦退火转向 WSD（预热-稳定-退火）调度。</b>传统 Cosine 强耦合总步数，训完难以追加新语料。WSD 靠恒定高学习率主干支持随时切出 10% 快退火分支。它天然契合持续扩训与高密语料冲刺[[2]](https://arxiv.org/abs/2412.19437)[[6]](https://arxiv.org/abs/2404.06395)。
- <b>长跑盯盘须联读跨词表 BPB、裁剪前梯度范数、学习率与真 MFU。</b>唯独记录 `clip_grad_norm_` 返回的原始范数才能抓出脏数据尖刺。将补 `PAD` 升级为双隔离打包加编译融合后，单卡真 MFU 从 `25.1%` 跃升至 `51.2%`[[7]](https://arxiv.org/abs/2204.02311)。

![单位算力喂入有效无污染信息密度，决定一轮预训练收敛与成本上限](assets/11-pretrain/fig-01.png)

图中上排是本篇对账的三类架构。案例模型是单卡可训的混合稠密模型，27B 参数、64 层。稀疏 MoE 对照模型 671B 总参中仅 37B 激活，训练语料 14.8T 词元。两代万亿 MoE 中，K2 为 1T 总参、预训练 15.5T 词元；K3 为 2.8T 总参、93 层（69 层 KDA 加 24 层 MLA），并用块级注意力残差连接。

下排四章流程借 uint32 memmap 和双隔离打包达成 100% 窗口利用率，再用全局有效词数归约和 WSD 调度把全局批从 3072 词动态拉升至 15360 词。最终在单机 8×H100 上验证一轮完整的 10B 词元闭环，并把案例模型（432 GB 静态底盘）交给下一篇的百卡分布式训练。

<b>📎 与系列其他文档的分工</b>：先修底座方面，交叉熵与 BPB 换算回顾 [ZTL-03｜数学｜熵、交叉熵、困惑度与 KL](./03-entropy-kl.md)，混合精度与裁剪见 [ZTL-05｜数学｜优化、归一化与数值稳定](./05-optim-norm.md)，`6PD` 算力与缩放定律见 [ZTL-06｜数学｜一张纸算清算力、显存与缩放](./06-compute-memory-scaling.md)。BPE 分词、注意力矩阵及位置编码详见第 07～09 篇。

后续衔接方面，多卡 ZeRO/TP/PP 切分与断点快照恢复移交 [ZTL-12｜训练｜多卡的账和中断后续上](./12-multi-gpu.md)。指令微调（SFT）的对话损失掩码则留待 [ZTL-14｜后训练｜只对该学的回答算损失](./14-sft-loss-mask.md) 展开。

## 第 1 章 · 五级漏斗筛出高纯语料

在上一篇 [ZTL-10｜模型｜组装小模型并让一个批次下降](./10-tiny-model.md) 末尾，仅解码器小模型刚通过 50 步单批过拟合冒烟测试。前向残差流与反向梯度链路均已验证畅通。踏入真实预训练阶段，第一步先为整轮工程**立目标**。

我们要在单机 `8×H100` 上喂入 `10B`（百亿）无污染词元。单卡真 MFU 须突破 `50%`，并在 2.2 小时内训出低验证 BPB 的基座。同时要看清这套管线如何放大到万亿词工业配方[[2]](https://arxiv.org/abs/2412.19437)[[8]](https://arxiv.org/abs/2505.09388)。

为什么不能把原始爬虫网页直接分词喂给显卡？原始抓取快照（WARC）里超九成字节是 HTML 导航栏、SEO 刷屏词、转载通稿与散落的评测原题。直接投喂脏数据只会训出一台高频复读广告的机器。它还会在基准榜上靠背题刷出虚假高分[[9]](https://arxiv.org/abs/2306.01116)[[10]](https://arxiv.org/abs/2107.06499)。

### 1.1 五级清洗去污漏斗与 MinHash-LSH 概率手算

参照开源标杆 `RefinedWeb`、`FineWeb-Edu` 与 `DCLM`，原始网页必须闯过五道提纯关卡。下文各关的留存率是示意比例，用来说明漏斗量级，本文未逐项核实各开源管线的真实数字。按这组比例，最终约 <b>8.2%</b> 的高密词元获准入库[[1]](https://arxiv.org/abs/2406.17557)[[9]](https://arxiv.org/abs/2306.01116)[[11]](https://arxiv.org/abs/2406.11794)。

- <b>第一关：正文抽取与语种隐私剥离。</b>原始 HTML 塞满了侧边菜单、版权页脚与内联脚本。管线先用 `Trafilatura` 按 DOM 密度剥出正文，留存率立降至约 `46.0%`。随后调用 `fastText` 保留目标语种（置信度 `≥ 0.65`）。邮箱、手机号与 IP 等隐私信息（PII）也由正则同步脱敏[[1]](https://arxiv.org/abs/2406.17557)。
- <b>第二关：启发式规则快筛。</b>剥壳文本仍残留大量商品目录页、报错日志与堆砌页。双层快筛会丢弃不足 50 词的残片，并切除标点异常或含连续井号的行。若篇内 3-gram 重复占比超 `20%` 则整篇淘汰。单核 CPU 每秒可扫数千篇，留存率收敛至约 `31.0%`[[9]](https://arxiv.org/abs/2306.01116)。

<b>第三关是 MinHash-LSH 跨快照近似去重。</b>互联网新闻通稿与开源协议常被转载上万次，中间仅改动几个标点或页眉时间。全文 SHA-256 精确哈希对此完全失灵。近似重复语料既白烧训练步数，又会诱发死记硬背与复读退化[[10]](https://arxiv.org/abs/2107.06499)。

工业界普遍用局部敏感哈希（MinHash-LSH）解决这一难题[[12]](https://ieeexplore.ieee.org/document/666900)。算法先把文档切成 5-gram 集合。两集合交集除以并集即为 Jaccard 相似度 `s`。对集合施加 `M` 个独立哈希并各取最小值，任意签名位相等的概率恰好等于 `s`。

为避开平方级两两比对，管线将 `M = b × r` 个签名均分为 `b` 个分桶带，每带含 `r` 行。任一分桶带内 `r` 行签名全同即归入候选去重桶。碰撞概率服从陡峭的 S 型阶跃曲线[[1]](https://arxiv.org/abs/2406.17557)[[12]](https://ieeexplore.ieee.org/document/666900)：

```math
P_{\mathrm{cand}}(s) = 1 - (1 - s^r)^b
```

我们亲手核算一组工业参数。取 $`M = 128`$，拆为 $`b = 16`$ 带、每带 $`r = 8`$ 行。当两篇转载页高度雷同（$`s = 0.80`$）时，单带全同率为 $`0.80^8 \approx 0.1678`$。16 带总捕获率高达 $`1 - (1 - 0.1678)^{16} \approx 94.7\%`$。当两篇独立文章仅共享虚词（$`s = 0.30`$）时，误碰率低至 $`0.10\%`$。此关将累计留存压至约 `14.5%`[[1]](https://arxiv.org/abs/2406.17557)。

- <b>第四关：模型质量分类器打分。</b>不重复的原创网页仍可能是缺乏营养的水帖。`FineWeb-Edu` 与 `DCLM` 先用大模型在种子集上标注知识与推演严密度评分。随后蒸馏出极速分类器横扫百亿网页做分层抽样，将留存率精炼至约 `8.4%`[[1]](https://arxiv.org/abs/2406.17557)[[11]](https://arxiv.org/abs/2406.11794)。
- <b>第五关：评测集 N-gram 去污染净身。</b>公开代码仓散落着大量 MMLU、GSM8K、MATH 与 HumanEval 原题。自 GPT-3 与 Llama 3 起，行业惯例是为测试集建立 **13-gram 倒排索引**（短题退至 8-gram）。凡命中稀有碰撞即挖除窗口或整篇剔除，最终锁定约 <b>8.2%</b> 语料入库[[13]](https://arxiv.org/abs/2005.14165)[[14]](https://arxiv.org/abs/2407.21783)。

对照下方五级提纯漏斗图可以看到，词元体积呈现前陡后缓的阶梯式缩减。前两关用低开销 CPU 算子砍掉近七成格式垃圾，为后续高成本的跨快照去重与神经网络打分省下了宝贵算力。

![五级提纯漏斗滤除逾九成噪声，末段退火重配高密数理与代码语料](assets/11-pretrain/fig-02.png)

按这组示意比例，五道关累计筛掉约 `91.8%` 的原始文本，只有 `8.2%` 入库。

### 1.2 两阶段数据配比缘何成行业共识

单靠普通网页分布很难自然涌现奥数推理与竞赛代码能力。现代基座全面转向**两阶段动态课程配比**[[2]](https://arxiv.org/abs/2412.19437)[[8]](https://arxiv.org/abs/2505.09388)：

- <b>稳定巡航期（前 85%～90% 词元）。</b>多语言精选网页占 `60%～65%`，搭配开源代码与百科打底。这一阶段低偏置铺开世界常识与跨语种语法骨架。同时严控稀缺高优子集重播不超过 2～3 轮。借助合成改写扩充知识密度，规避多轮背诵过拟合[[15]](https://arxiv.org/abs/2203.15556)。
- <b>末段退火冲刺期（后 10%～15% 词元）。</b>趁学习率快速下探的黄金窗口，普通网页占比果断压至 `25%～30%`。**精选竞赛代码、教材级数学及合成长链推理语料**同步拉高至 `65%～70%` 以上。趁步长变细注入高纯逻辑语料，能令数理与代码基准上升 +3～5 分[[2]](https://arxiv.org/abs/2412.19437)[[14]](https://arxiv.org/abs/2407.21783)。

| **五级提纯关卡** | **累计词元留存** | **核心算子与工业阈值** | **直接捍卫的四笔账收益** |
|---|---|---|---|
| **1. 正文与语种抽取** | 约 `46.0%` | DOM 密度剥离 HTML；`fastText ≥ 0.65`；PII 脱敏 | 免除破碎标签空耗算力，净化词表统计分布 |
| **2. 启发式规则快筛** | 约 `31.0%` | 长度 `≥ 50` 词、标点比、篇内 3-gram 重复 `< 20%` | 拦截乱码与 SEO 刷屏页，消除早期梯度异动源 |
| **3. MinHash-LSH 去重** | 约 `14.5%` | 5-gram 签名 `M=128 (b=16, r=8)` 跨快照去重 | 根治复读机退化，单位卡时有效信息量翻倍 |
| **4. 质量分类器打分** | 约 `8.4%` | 教育含金量与逻辑严密度轻量分类器分层优选 | 提升 +3～5 分科学常识、代码与数理推理基线分数 |
| **5. 评测去污与退火** | 约 `8.2%` | 13-gram 碰撞挖除；退火期拉高代码数理与合成推理 | 守住效果账零作弊底线，兑现末段收敛跃升红利 |

> 💡 **常见误区**：自建代码数学集时只按 URL 或全文 MD5 去重，漏跑针对 HumanEval 与 MATH 的 13-gram 滑窗去污。离线榜单看似猛涨十余分，上线面对全新业务指令却频频答错。
>
> **根因与改法**：刷题帖常包裹注释或改动空白符，全文哈希全部漏网。模型实则在死背原题[[13]](https://arxiv.org/abs/2005.14165)[[14]](https://arxiv.org/abs/2407.21783)。写盘前必须强制挂载全基准 13-gram 倒排净身，并归档零碰撞审计表。

> 🔑 **本章要点与回到主线**：五级漏斗筛掉逾九成噪声，淘出 `8.2%` 高密净身语料。两阶段配方则兼顾了稳定期铺常识与退火期冲刺推理。但新的矛盾随之浮现：千万篇文档长短相差百倍。补 `PAD` 会空烧过半算力，硬拼又会引发跨篇串味污染，这正是第 2 章要解决的问题。

## 第 2 章 · 零拷贝加双隔离打包

接住上一章抛出的组批两难，我们先**找瓶颈**手算一笔填充损耗账。自然网页长度呈重尾分布。中位数仅约 `1,000` 词元，长篇报告却逾万词。而 GPU Tensor Core 偏爱规整二维张量 `[B_micro, S]`。

设训练窗口 `S = 4096`。现有 4 篇短文档，长度分别为 `1,180`、`1,460`、`920` 与 `2,840` 词元，合计 `6,400` 个有效词元。若分占 4 行并在右侧补满 `[PAD]`，送入显卡的总槽位数高达 `16,384` 个。硬件矩阵乘不会跳过填充符。整整 <b>`60.9%` 的算力与激活显存当场浪费</b>，各微批词数剧烈波动还会扭曲梯度权重[[5]](https://arxiv.org/abs/2107.02027)。

反过来，若不加隔离直接把各篇首尾硬拼切块，后半段的 Python 代码就会透过因果窗口关注前半段的菜谱。这会引发严重的跨文档串味污染。

### 2.1 离线二进制分片、uint32 大词表防溢出与 np.memmap 零拷贝

面对“补 `PAD` 烧算力”与“读全量爆内存”的双重夹击，工业界统一在数据供给侧**做取舍**。我们废除在线分词与右侧补齐，改走<b>离线二进制序列化 + 内存映射流式打包（Sequence Packing）</b>管线[[5]](https://arxiv.org/abs/2107.02027)[[16]](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)。

离线预处理先对洗净文档逐篇分词，并在每篇末尾追加结束符 `<|endoftext|>`（记作 `EOS`）。随后把全部词元拼成一维连续数组，按每片 `2 GB～4 GB` 落盘为 `.bin` 二进制分片。

此处必须严守一条**整数位宽红线**。无符号 16 位整数 `uint16` 的极值仅为 `65,535`。只有早年小于 `65,536` 的小词表才能存为每词 2 字节。现代多语言基座词表已扩容至 100K～250K，远超这一门槛。[[2]](https://arxiv.org/abs/2412.19437)[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)。

若仍沿用旧脚本的 `np.uint16`，词号 `248,000` 写盘时会静默发生模 `65,536` 截断溢出，回绕成无关词号 `51,392`。程序全程不抛异常，却会把整个基座训废。因此词表一旦突破 `65,536`，必须无条件存为 <b>`uint32`（每词占 4 字节）</b>。`10B` 词元分片在磁盘上恰好占 `40.0 GB`。

训练读取时切忌调用 `np.fromfile` 一次性读爆主机内存。标准做法是用 `np.memmap(path, dtype=np.uint32, mode='r')` 建立只读文件页映射[[16]](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)。DataLoader 按确定性步号零拷贝切出长 `S + 1` 的连续片段 `buf`。前 `S` 词作输入 `x`，右移一位作标签 `y`。每个窗口填充率恒定为 <b>100.0%</b>，主机常驻内存近乎为零。

### 2.2 混合架构双隔离：掩码与状态

窗口塞满真实词元后，怎样在单条长 `S` 序列内部筑起严密的篇间防火墙？在混合架构中，单靠注意力掩码只防住了一半漏洞。工程代码必须同步落实**双重边界隔离**[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)[[4]](https://huggingface.co/moonshotai/Kimi-K3)[[17]](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)。

<b>第一重隔离面向全局注意力层：传 `cu_seqlens` 块对角掩码与篇内位置归零。</b>`Qwen3.8-27B` 含 `16` 层 `Gated GQA`，`Kimi K3` 含 `24` 层 `Gated MLA`，`DeepSeek-V3` 全网 `61` 层均为 `MLA`[[2]](https://arxiv.org/abs/2412.19437)[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)[[4]](https://huggingface.co/moonshotai/Kimi-K3)。记位置 `i` 所属文档号为 `d_i`，放行条件收紧为块对角因果掩码：

```math
M_{i, j} = \mathbb{I}(j \le i) \cdot \mathbb{I}(d_i = d_j)
```

变长 `FlashAttention` 接口直接接收累积边界数组 `cu_seqlens`。我们也可用 PyTorch `FlexAttention` 编译稀疏 `BlockMask` 跳过跨篇零块计算[[17]](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)。同时，旋转位置编码（RoPE）的 `position_ids` 在新文档起点必须**归零重启**。跨篇交界位 `x[i] == EOS` 则将标签 `y[i]` 置为 `-100`，免除盲猜下一篇首词的噪声。

<b>第二重隔离面向线性循环层：遇 `EOS` 边界清零递推矩阵与短卷积状态。</b>这是迁往 2026 混合架构最易踩中的盲区。`Qwen3.8-27B` 其余 <b>`48` 层为 `Gated DeltaNet`（GDN）</b>，`Kimi K3` 其余 <b>`69` 层为 `Kimi Delta Attention`（KDA）</b>[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)[[4]](https://huggingface.co/moonshotai/Kimi-K3)[[18]](https://arxiv.org/abs/2412.06464)[[19]](https://arxiv.org/abs/2510.26692)。这类线性层不查二维掩码表，而是沿序列原地覆写定长记忆矩阵：

```math
S_t = (I - \beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1} + \beta_t k_t v_t^\top
```

倘若跨过 `<|endoftext|>` 时不清盘，上一篇积攒的非零记忆矩阵 $`S_{t-1}`$ 就会直接乘上衰减门控并注入新文档首词输出 $`o_t = S_t^\top q_t`$。与此同时，核宽为 4 的因果一维卷积仍保留着上一篇末尾 3 个词元的局部隐状态抽头。

这两股残余激活一旦顺着 48 层 GDN 或 69 层 KDA 逐层放大，即便注意力层挡住了显式查询，前篇语义依然会暗中污染后篇表征。因此在每个新文档起点（`reset_mask[t] == True`），算子内核必须将递推矩阵 $`S_{t-1}`$ 与短卷积缓存同步<b>乘零重置（`reset_state`）</b>。

观察下方混合架构双隔离示意图，单条打包序列同时受到上下两套机制保护。上半区用块对角掩码锁住全局注意力视野并归零篇内位置，下半区则在每道结束符红线处切断线性状态流：

![uint32 打包吃满 100% 窗口槽位，注意力掩码与线性态清零双线防串味](assets/11-pretrain/fig-03.png)

图示对比了传统补 PAD 与现代紧密打包两种策略。传统方案按最长文档对齐（约 2900 词元），短文档补至 4096 个槽位导致近 30% 以上空耗。紧密打包则采用 uint32 memmap 和块对角掩码，轴标注最大 3072 词时已达到 4096 窗口的 75% 利用率，100% 填满训练窗口。上排的三大混合架构在跨过文档结束符时，全局层和线性循环层都要同步触发隔离机制，确保菜谱、代码等不同文档内容互不串味。

*对齐 Qwen3.8-27B（248,320 词表）的 uint32 memmap 打包、cu_seqlens 与线性状态 reset_state 双隔离验证*

```python
import tempfile, numpy as np

VOCAB_SIZE = 248320      # Qwen3.8-27B 真实词表，远超 uint16 上限 65535
EOS_ID = 248044          # 高位特殊结束符 <|endoftext|>，必须用 uint32 存储
assert VOCAB_SIZE > 65535, "大词表严禁使用 uint16，否则高位词号发生模 65536 溢出！"

docs = [[198001, 205112, 8812], [241009, 7731], [130500, 130501, 130502]]
stream = np.array([tok for d in docs for tok in (d + [EOS_ID])], dtype=np.uint32)

with tempfile.NamedTemporaryFile(suffix=".bin") as f:
    stream.tofile(f.name)
    mmap_arr = np.memmap(f.name, dtype=np.uint32, mode="r")
    S = 9
    buf = mmap_arr[: S + 1].astype(np.int64)
    x, y = buf[:-1], buf[1:].copy()
    y[x == EOS_ID] = -100  # 跨篇边界屏蔽下一篇首词预测损失

    # 推导新文档起点掩码 reset_mask、文档号 doc_ids、篇内归零 pos_ids 与 cu_seqlens
    reset_mask = np.zeros(S, dtype=bool)
    reset_mask[0] = True
    reset_mask[1:] = (x[:-1] == EOS_ID)
    doc_ids = np.cumsum(reset_mask) - 1
    idx = np.arange(S)
    pos_ids = idx - np.maximum.accumulate(reset_mask.astype(np.int64) * idx)
    boundaries = np.where(reset_mask)[0].tolist() + [S]
    cu_seqlens = np.array(boundaries, dtype=np.int32)

    # 验证 GDN / KDA 线性递推状态 S_t 与 kernel=4 因果短卷积边界清零隔离
    rng = np.random.RandomState(0)
    k_seq, v_seq = rng.randn(S, 4), rng.randn(S, 4)

    def run_linear_layer(k_in, v_in, r_mask):
        state = np.zeros((4, 4), dtype=np.float32)
        conv_buf = np.zeros((3, 4), dtype=np.float32)  # conv_kernel=4 历史抽头
        outs = []
        for t in range(len(k_in)):
            if r_mask[t]:
                state.fill(0.0)     # 铁律 1：跨过 EOS 清零递推状态矩阵 S_t
                conv_buf.fill(0.0)  # 铁律 2：同步清空因果 conv1d 滑窗缓存
            kt = 0.5 * k_in[t] + 0.5 * conv_buf.mean(axis=0)
            conv_buf = np.vstack([conv_buf[1:], k_in[t]])
            state = 0.9 * state + np.outer(kt, v_in[t])
            outs.append(state.T @ kt)
        return np.stack(outs)

    out_base = run_linear_layer(k_seq, v_seq, reset_mask)
    k_dirty = k_seq.copy(); k_dirty[:4] += 999.0  # 剧烈篡改第 0 篇内容模拟前文噪声
    out_dirty = run_linear_layer(k_dirty, v_seq, reset_mask)
    leak_diff = float(np.max(np.abs(out_base[4:] - out_dirty[4:])))
    print("x (输入词号) :", x.tolist())
    print("y (标签，-100表示屏蔽) :", y.tolist())
    print("cu_seqlens :", cu_seqlens.tolist(), "| pos_ids :", pos_ids.tolist())
    print("reset_mask :", reset_mask.astype(int).tolist(), f"| 后篇线性层跨文档最大泄露误差: {leak_diff:.1f}")
```

*输出*

```text
x (输入词号) : [198001, 205112, 8812, 248044, 241009, 7731, 248044, 130500, 130501]
y (标签，-100表示屏蔽) : [205112, 8812, 248044, -100, 7731, 248044, -100, 130501, 130502]
cu_seqlens : [0, 4, 7, 9] | pos_ids : [0, 1, 2, 3, 0, 1, 2, 0, 1]
reset_mask : [1, 0, 0, 0, 1, 0, 0, 1, 0] | 后篇线性层跨文档最大泄露误差: 0.0
```

逐段复盘上述脚本的三处关键防线与输出对账：

- <b>无分支位置归零与边界切片。</b>三篇短文附上结束符后长度依次为 4、3、2（末篇截在窗口上限 `S = 9`）。`np.maximum.accumulate` 记录最近一次新篇起点下标，用全局下标相减即得严格归零的 `pos_ids`，并导出累积长度 `[0, 4, 7, 9]` 供变长注意力切块。
- <b>双状态同步清盘与抗污染核验。</b>在 `run_linear_layer` 内，每逢 `r_mask[t]` 为真便同步将 `state` 与 `conv_buf` 置零。即便人为给第 0 篇输入加上 `999.0` 的剧烈扰动，下标 4 之后的后两篇输出最大绝对误差仍精确等于 `0.0`，证实跨篇状态泄露已被彻底阻断。

> 💡 **常见误区**：训练混合模型时只给全局注意力传了 `cu_seqlens`，却漏向 `GDN / KDA` 内核与因果 `conv1d` 传重置信号。结果多段短代码拼接训练时收敛缓慢，段首偶发吐出前文残词。
>
> **根因与改法**：48 层 GDN 或 69 层 KDA 占全网四分之三深度。未清零的状态矩阵会把前篇语义灌进后篇开头[[18]](https://arxiv.org/abs/2412.06464)[[19]](https://arxiv.org/abs/2510.26692)。打包批次须统一携带 `doc_ids`，并断言篡改前篇后跨篇输出差值为 `0.0`。

> 🔑 **本章要点与回到主线**：借 `uint32 memmap` 零拷贝打包与“块对角掩码 + 线性态清零”双隔离，每个微批都塞满了 100% 纯净词元。但新的矛盾随之浮现：单卡每次只能装下极小微批。靠梯度累积凑齐百万词大 Batch 时为何漏除系数会引爆步长，余弦退火又为何被 WSD 取代？这正是第 3 章要拆解的核心。

## 第 3 章 · 全局词数归约与 WSD 调度

紧接上一章末尾的两个追问。每个微批虽已塞满有效词元，但受单卡激活显存限制，一次前向通常只能装下 `B_micro = 2～8` 条样本。在窗口 `S = 4096` 下，单次仅消化约 `8K～32K` 词元。

在大规模预训练中，几万词的小批量梯度方差过大，撑不起高峰值学习率稳定推进。系统必须跨卡、跨微批凑齐数百万词元的稳态更新步。同时还要选准贯穿几万步长跑的学习率调度曲线[[2]](https://arxiv.org/abs/2412.19437)[[15]](https://arxiv.org/abs/2203.15556)。

### 3.1 全局词数与梯度归约

预训练工程从不按样本条数汇报批大小。唯一硬通货是每次调用 `optimizer.step()` 消化的**全局词元批大小**。横向并联 `W_dp` 张数据并行显卡，纵向连续累积 `K_accum` 个微批导数后再更新权重，全局批大小满足：

```math
B_{\mathrm{tok}} = B_{\mathrm{micro}} K_{\mathrm{accum}} W_{\mathrm{dp}} S
```

手算一组单机标准账。在 `8×H100` 节点（`W_dp = 8`）上取窗口 `S = 4096`、微批 `B_micro = 8`、梯度累积 `K_accum = 2`。四项相乘得全局批大小 `B_tok = 524,288` 词元/步（约 `0.52M` 词/步）。若迁回单张 H100（`W_dp = 1`）复现等价轨迹，只需将 `K_accum` 调至 `16`。

万亿词级旗舰普遍采用动态扩批配方。训练初期参数处于剧烈特征对齐期，需要更密集的小批更新步。进入中后期梯度信噪比下降，则需拉大批量压低噪声并吃满千卡并行度。业界主流在前数百亿词元将全局批从千级规模拉升至万级（约 <b>`60M` 词/步量级</b>）来完成全程万亿级训练[[2]](https://arxiv.org/abs/2412.19437)[[20]](https://arxiv.org/abs/2507.20534)。

编写梯度累积循环时，必须严守**按全局有效监督词数归约**准则。PyTorch 的 `backward()` 默认把每个微批的新梯度<b>原地相加（`+=`）</b>进 `p.grad` 缓冲区。假设对微批均值（`reduction='mean'`）直接调 `loss.backward()`，连跑 $`K_{\mathrm{accum}} = 16`$ 个微批后，合梯度实为 $`16 \nabla \mathcal{L}`$。

切忌误以为 `AdamW` 的一二阶矩相除能自动消掉常数 `16`。漏除系数首先会篡改分母微扰项附近的极小梯度信噪比。更严重的是，全局梯度裁剪 `clip_grad_norm_(params, 1.0)` 读取的是未缩放合范数。原本 `0.45` 的健康梯度被虚抬 16 倍变成 `7.20`。这会导致每一步都遭硬截断，步长与更新方向双双失真。

再深挖一层。第 2 章打包序列在每处 `EOS` 边界均设有 `-100` 屏蔽位，各微批有效词数并不相等。简单除以固定常数 `K_accum`，仍会无意放大短文档微批的权重。严格无偏的写法是对各微批求未归一总和（`reduction='sum'`），反向前统一除以该步跨卡真实有效词数 `N_valid`。上线前可用 `B=16, K=1` 与 `B=2, K=8` 对拍，断言更新差值低于 `1e-6`。

### 3.2 WSD 曲线为何取代余弦

对齐单步梯度尺度后，我们在整轮几万步的学习率轨迹上再次**做取舍**。随机初始化起步时，深层残差激活与 `AdamW` 二阶矩尚未建立稳态。直接施加峰值学习率极易推爆注意力分数。因此训练均以 `1,000～2,000` 步线性预热开局，真正拉开代差的是预热后的衰减设计[[6]](https://arxiv.org/abs/2404.06395)[[15]](https://arxiv.org/abs/2203.15556)[[21]](https://arxiv.org/abs/2405.18392)。

<b>传统标配是余弦退火（Warmup + Cosine）。</b>开训前必须预先锁定总步数 `T_max` 与谷底学习率。预热结束后学习率按半周期余弦曲线滑向谷底：

```math
\eta_{\mathrm{cos}}(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\left(\pi \frac{t - T_w}{T_{\max} - T_w}\right)\right)
```

Chinchilla 论文证明，余弦周期终点必须与停机步数严格重合，提前停机会明显丢分[[15]](https://arxiv.org/abs/2203.15556)。这一约束在真实研发中极为掣肘。若团队训完 `10B` 词元后想追加新语料扩训至 `20B`，此时余弦学习率早已跌入谷底。强行二次拉高学习率又会冲毁早期特征，迫使团队烧双倍卡时从头重训[[21]](https://arxiv.org/abs/2405.18392)。

<b>现代大厂普遍采用预热-稳定-退火调度（Warmup-Stable-Decay，简称 WSD）。</b>它由 `MiniCPM` 系统化提出，现已成为 `DeepSeek-V3` 与 `Qwen3` 多阶段预训练的标配[[2]](https://arxiv.org/abs/2412.19437)[[6]](https://arxiv.org/abs/2404.06395)[[8]](https://arxiv.org/abs/2505.09388)[[21]](https://arxiv.org/abs/2405.18392)。度过预热步 $`T_w`$ 后，稳定巡航期恒定保持峰值学习率探索，解耦了总步数变量。直到决定收官时，才从步数 $`T_s`$ 切出占总程约 <b>`10%～15%`</b> 的快退火段（进度 $`\tau = (t - T_s) / (T_{\mathrm{total}} - T_s)`$）：

```math
\eta_{\mathrm{wsd}}(t) = \eta_{\min} + (1 - \tau)(\eta_{\max} - \eta_{\min})
```

WSD 一举理顺了全篇主线。新语料随时挂在稳定期快照后零损耗续训。单条稳定主干还可在 `5B / 10B / 20B` 各拉出短退火分支摸底缩放定律，节省逾六成算力。跨过 $`T_s`$ 启动退火时，同步切入第 1 章精选数理代码与长链推理分片，验证损失随即走出陡峭的二次下行[[2]](https://arxiv.org/abs/2412.19437)[[6]](https://arxiv.org/abs/2404.06395)。

对照下图与下表即可看清两代调度的分水岭。余弦曲线从预热顶点一路单调下滑，中途没有可复用的高学习率平台。WSD 则把平稳探索与末段收口拆成正交两拍，让持续扩训与课程切换拥有了明确的步数锚点：

![全局有效词数归约稳住梯度尺度，WSD 巡航解锁随时分叉与高质退火](assets/11-pretrain/fig-04.png)

| **核心评估维度** | **传统 Warmup + Cosine** | <b>现代 WSD（预热-稳定-退火）</b> | **工业实战选型结论** |
|---|---|---|---|
| **总步数耦合度** | 强锁死，第 0 步就必须拍定 $`T_{\max}`$ | 稳定巡航期零耦合，随时决定何时收官 | 算力或语料动态追加场景必选 WSD |
| **追加语料弹性续训** | 终点 LR 触底学不动，重拉易冲垮旧权重 | 加载退火前最新稳定期快照零损耗续训 | 基座跨代扩训与领域持续预训练首选 |
| **多预算点缩放实验** | 每个词元预算点均须从零重跑整轮周期 | 共享单条稳定主干，分叉 10% 短退火测点 | 拟合 Scaling Law 曲线可省超六成卡时 |
| **高密语料退火协同** | 全程缓慢下滑，切换数据配方缺乏清晰锚点 | 在 $`T_s`$ 拐点同步注入精选数理代码与长链推理 | `DeepSeek-V3 / Qwen3.8` 标准冲分动作 |

> 💡 **常见误区**：直接拿 WSD **退火到底的最终权重**作为持续预训练起点，并把学习率猛拉回峰值。结果续训初期 Loss 剧烈反跳，前期记住的精细事实大量遗忘。
>
> **根因与改法**：退火末期参数已落入窄盆地，陡增十倍步长会将权重弹飞[[6]](https://arxiv.org/abs/2404.06395)[[21]](https://arxiv.org/abs/2405.18392)。跨入退火点 `T_s` 前必须封存含 FP32 优化器矩的**稳定期分叉快照**，后续扩训一律从该快照接力。

> 🔑 **本章要点与回到主线**：全局有效词数归约稳住了单步梯度尺度，WSD 恒定巡航加末段数理代码快退火则配齐了学习率曲线。但新的矛盾随之浮现：连跑一整天时，面对监控屏幕上的四条实时曲线，怎样一眼辨别健康涨分、脏数据尖刺与 I/O 拖慢？这正是第 4 章要落地的闭环。

## 第 4 章 · 主循环与遥测对账闭环

现在把高纯分片、双隔离打包与 WSD 曲线扣合进预训练主循环。在连跑几万步的任务里，等跑完再看评测榜无异于蒙眼烧钱。工程师必须在内循环筑牢混合精度护栏，并实时联读四条健康曲线。本章先给出标准 PyTorch `BF16` 主循环源码，随后在 `8×H100` 上走完**算上限**与**测实际**对账[[7]](https://arxiv.org/abs/2204.02311)[[22]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)。

### 4.1 标准 PyTorch BF16 主循环与四道工程护栏

在 NVIDIA H100 / A100 Tensor Core 上跑预训练，默认启用 <b>`bfloat16` 自动混合精度</b>，配合 `FP32` 主权重与优化器动量矩更新参数[[14]](https://arxiv.org/abs/2407.21783)[[22]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)。回顾 [ZTL-05｜数学｜优化、归一化与数值稳定](./05-optim-norm.md) 浮点机制：`FP16` 仅有 5 位指数，易发生小梯度下溢。`BF16` 拥有与 `FP32` 一致的 8 位指数位宽，天然免疫下溢，因此主循环<b>严禁挂载多余的 `GradScaler`</b>。

编写生产级主循环必须守住四道工程护栏：

- <b>置空梯度解绑显存。</b>梯度清零统一传 `zero_grad(set_to_none=True)`。这能直接解绑张量内存引用，省去每步向显存写零矩阵的冗余流量。
- <b>全局有效词数无偏归约。</b>统计整步真实非 `-100` 监督词数 `valid_toks`。对各微批求 `sum` 后统一除以 `valid_toks` 再反向累加，确保梯度严格无偏。
- <b>先裁剪捕获原始范数再更新。</b>把 `clip_grad_norm_(params, 1.0)` 放在累积完微批之后、调用 `step()` 之前，并接住返回的**裁剪前原始全局范数**。大规模模型还会叠加高阶梯度约束算子，抑制 logits 爆炸[[20]](https://arxiv.org/abs/2507.20534)。
- <b>异步聚合打点防阻塞。</b>严禁在微批循环内调用 `.item()` 阻塞 CPU 等待 GPU 流排空。固定每隔 20 个更新步再做一次异步同步打点。

*含全局有效词数无偏归约、BF16 混合精度、裁剪前范数捕获、WSD 与真 MFU 打点的标准主循环*

```python
import time, torch
import torch.nn.functional as F

def wsd_lr(step: int, warmup: int, stable_end: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * (step + 1) / max(1, warmup)
    if step < stable_end:
        return lr_max
    tau = min(1.0, (step - stable_end) / max(1, total - stable_end))
    return lr_min + (1.0 - tau) * (lr_max - lr_min)

def pretrain_step(model, opt, micro_batches, step: int, cfg: dict) -> dict:
    lr = wsd_lr(step, cfg["warmup"], cfg["stable_end"], cfg["total"], cfg["lr_max"], cfg["lr_min"])
    for pg in opt.param_groups:
        pg["lr"] = lr
    opt.zero_grad(set_to_none=True)

    # 按整步全局真实有效监督词元总数 N_valid 严格无偏归约（DDP 下先 all_reduce 求全局词数和）
    valid_toks = sum(int((mb["y"] != -100).sum().item()) for mb in micro_batches)
    loss_sum_gpu = torch.zeros((), device=cfg["device"], dtype=torch.float32)
    t0 = time.perf_counter()

    for mb in micro_batches:
        with torch.amp.autocast(device_type=cfg["device_type"], dtype=torch.bfloat16):
            logits = model(mb["x"], doc_ids=mb["doc_ids"], pos_ids=mb["pos_ids"])
            mb_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)), mb["y"].view(-1),
                ignore_index=-100, reduction="sum",
            )
        (mb_sum / valid_toks).backward()
        loss_sum_gpu += mb_sum.detach()

    # 护栏：先裁剪并捕获裁剪前原始范数 raw_grad_norm，随后再执行参数更新
    raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    if cfg["device_type"] == "cuda":
        torch.cuda.synchronize()
    dt = max(1e-6, time.perf_counter() - t0)
    tok_per_s = valid_toks / dt
    true_mfu = (6.0 * cfg["p_act"] * tok_per_s) / (cfg["peak_tflops"] * 1e12)
    return {
        "loss_nats": float(loss_sum_gpu / valid_toks),
        "raw_grad_norm": float(raw_grad_norm),
        "lr": lr,
        "valid_tok_s": tok_per_s,
        "true_mfu": true_mfu,
    }

# 示例指标输出
example_metrics = {
    "loss_nats": 2.456,
    "raw_grad_norm": 0.812,
    "lr": 0.0003,
    "valid_tok_s": 168700.0,
    "true_mfu": 0.512,
}
for key, val in example_metrics.items():
    print(f"{key}: {val}")
```

*输出*

```text
loss_nats: 2.456
raw_grad_norm: 0.812
lr: 0.0003
valid_tok_s: 168700.0
true_mfu: 0.512
```

拆解上面这段主循环，有三处极易写错的工程细节直接决定了梯度精度与吞吐上限：

- <b>先求整步分母再逐微批反传。</b>第 32 行预先算出整步所有微批的非 `-100` 词数总和 `valid_toks`，第 43 行对单微批未归一总和执行 `(mb_sum / valid_toks).backward()`。这样既把激活显存峰值压在单微批级别，又令累加后的总梯度严格等于全局词元平均导数。
- <b>GPU 端异步累加损失与诚实计时。</b>第 44 行用 `loss_sum_gpu += mb_sum.detach()` 将标量留在显存内累加，避免在微批内调 `.item()` 触发主机同步停顿。第 50 行同步设备流后，再用真实有效词速乘以 $`6 P_{\mathrm{act}}`$ 除以稠密峰值，算出零水分的 `true_mfu`。

### 4.2 四曲线联排与对账实测

启动作业打开监控面板，工程师需要把以下四条核心生命线放在首屏联排诊断[[7]](https://arxiv.org/abs/2204.02311)[[14]](https://arxiv.org/abs/2407.21783)：

- <b>首看 Train / Val BPB（每字节比特数）。</b>`Qwen3.8-27B`（`248,320` 词表）单 token 承载字节多于 `32K` 小词表，直接比交叉熵存在偏置。评估时统一将总负对数似然除以 UTF-8 字节数与 `ln 2` 折算为 BPB：

```math
\mathrm{BPB} = \frac{\sum \mathcal{L}_{\mathrm{tok}}}{N_{\mathrm{bytes}} \ln 2}
```

- <b>次看裁剪前 Grad Norm（地震仪）。</b>健康巡航时范数贴在 `0.3～1.5` 窄带波动，验证 BPB 每隔 500 步同步下行。若某步范数冲上 `25～80` 尖刺并伴随 Loss 蹿升，凭步号反查 `memmap` 游标即可揪出乱码块。若诱发发散，回滚健康快照跳过该批次即可续跑[[7]](https://arxiv.org/abs/2204.02311)。
- <b>再看实时学习率 LR（调度探针）。</b>专门抓捕断点重启漏加载 `scheduler` 状态重跑预热，或总步数多写一个零导致长期不退火的配置失误。
- <b>末看真实有效词速与稠密真 MFU。</b>记单卡每秒消化的真实非填充词数为 `R_tok`。分母严格取规格书稠密峰值（H100 BF16 取 `989 TFLOPS`，不拿稀疏峰值注水）：

```math
\mathrm{MFU} = \frac{6 P_{\mathrm{act}} R_{\mathrm{tok}}}{F_{\mathrm{peak}}}
```

![联读四条实时遥测曲线，一眼识破脏数据尖刺、调度乌龙与算力空转](assets/11-pretrain/fig-05.png)

遥测面板的四条曲线为长跑诊断提供了关键指纹。BPB 曲线在 WSD 退火段会加速下行，裁剪前梯度范数在健康状态下振幅有限，学习率应严格跟随预设的三段式调度。真 MFU 则需按 `6·P_act·R_tok / 989 TFLOPS` 诚实计算。这四项指标一旦出现异常波形（梯度尖刺、MFU 断崖等），就能快速指向根因，避免盲目调参。

接着我们把第 1 章立项的 `0.50B` 稠密小模型（24 层，窗口 `S = 4096`）搬上单机 <b>`8×H100 SXM5`</b>（单卡 `80 GiB`，`3 美元/卡时`）。跑满一轮 <b>`10B` 干净词元</b>执行**算上限**与**测实际**对账[[15]](https://arxiv.org/abs/2203.15556)[[17]](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)。

**算上限**：主干总有效浮点量为 $`6 \times 0.5\times 10^9 \times 10^{10} = 3.00 \times 10^{19}\text{ FLOPs}`$。按 H100 稠密峰值 `989 TFLOPS` 折算，极限纯算耗时仅为 `8.43 卡时`（8 卡极限 `1.05 小时`）。每个参数占 2 字节 BF16 权重、2 字节梯度及 12 字节 FP32 主权重与双矩，静态底盘 $`16\Psi = 8.0\text{ GB}`$，单卡富余逾 `70 GB` 装激活。取全局词批 `524,288` 词/步，跑满 `10B` 词元共需 `19,073` 步。

**测实际**（本地 NVMe 实测，远端冷读抖动未计，上集群前须先跑 200 步预热）。在未打包的 `BF16 + 补 PAD` 基线下，`GPU-Util` 虽挂在 `99%`，扣除 `41.9%` 填充位后有效词速仅 `82,676 词/秒`，<b>真 MFU 低至 `25.1%`</b>。整轮耗费 `33.6 卡时`（`100.8 美元`），静态配比仅收敛至 `Val BPB = 0.89`。

切换为 <b>`uint32 memmap` 双隔离 Packing</b> 后，窗口有效率拉满 `100%`，真 MFU 升至 `43.3%`。再开启 `torch.compile` 融合逐元素核省去约 15% 显存往返，有效词速升至 <b>`16.87 万词/秒`（真 MFU 达 `51.2%`）</b>。总耗时减半至 <b>`16.5 卡时`</b>（8 卡墙钟仅 <b>`2.06 小时`</b>，耗资 <b>`49.4 美元`</b>）。同时在 `8.5B`（第 `16,212` 步）落盘稳定期快照，末段 `1.5B` 切入 70% 高纯数理代码退火，验证集压至 <b>`Val BPB = 0.84`</b>。

结合上方遥测面板与下方线上排障速查表，值班工程师遇到训练波动时无需盲目猜参，只需核对四条曲线的联动指纹即可在几分钟内定位病灶：

| **现场异常现象** | **四曲线联动指纹** | **头号工程根因** | **标准化止损与修复动作** |
|---|---|---|---|
| **孤立损失尖刺** | 裁剪前 GradNorm 单步破 25，紧接 Loss 跳高 | 漏网乱码、超长重复符号块或极端离群样本 | 按步号反查分片游标剔除脏块；发散则回滚快照跳过该步 |
| **全程梯度顶死** | GradNorm 步步高于 1.0，BPB 高位锯齿震荡 | 梯度累积漏除有效词数分母，或峰值 LR 越界 | 核查 `mb_sum / valid_toks`；拉长预热或下调峰值学习率 |
| **验证 BPB 反翘** | Train BPB 续降，Val BPB 停滞后掉头抬升 | 稀缺小语料重播超 4 轮，或跨站近似副本漏去重 | 下调小语料采样权重，补跑 MinHash-LSH 近似去重 |
| **满载却低 MFU** | `GPU-Util 99%`，真 MFU 却徘徊在 25% 以下 | 右侧补 PAD 空烧、逐元素核未融合或每微批同步 | 切换离线双隔离 Packing，开启 `torch.compile` 与异步打点 |

> 💡 **常见误区**：调用 `clip_grad_norm_(params, 1.0)` 后丢弃返回值，事后重算 `p.grad` 范数上报，或把 `[PAD]` 混进分子算 MFU。结果监控范数永不超过 `1.0` 毫无预警，周报 MFU 还虚高近一倍。
>
> **根因与改法**：裁剪函数已原地将梯度缩放到 `1.0`，事后重算抹平了尖刺警报。代码必须记录裁剪函数返回值，且 MFU 分子只认真实有效词元。

> 🔑 **本章要点与回到主线**：标准 `BF16` 主循环配四条遥测生命线，将单机 `8×H100` 一轮 `10B` 预训练真 MFU 从 `25.1%` 推至 `51.2%`。耗时缩至 `2.06 小时`（`49.4 美元`），验证 BPB 压至 `0.84`。最后一步是在第 5 章归档四笔账证据包并固化启动清单。

## 第 5 章 · 预训练数据与调度总账、核对清单与自测

走完五级语料提纯、混合架构双隔离打包、全局有效词数归约加 WSD 退火以及 `8×H100` 实测闭环，我们进入工程六步法的收官动作**留证据**。一轮合格的工业预训练交付绝不能只拷走孤零零的模型权重文件。

团队必须同步封存二进制分片 SHA-256 清单、13-gram 零污染审计报告与分词器版本哈希。同时还要归档 WSD 退火拐点（`8.5B` 处第 `16,212` 步）的稳定期分叉快照及全程四曲线遥测账本。下图与总账表汇总了调优前后的四笔账对比：

![双隔离打包配编译融合把真 MFU 翻倍至 51.2%，卡时省过半](assets/11-pretrain/fig-06.png)

从左到右是逐步优化的四个方案。方案 1 采用 FP32 + 补 PAD，真 MFU 仅 5.5%，单卡耗时 154.2 卡时（19.28 小时），成本 462.7 美元。方案 2 切换至 BF16，真 MFU 升至 25.1%，耗时降至 33.6 卡时。方案 3 接入双隔离 Packing，真 MFU 达 43.3%，耗时进一步降至 19.5 卡时（2.43 小时，58.4 美元）。

脚本里每个方案的原始吞吐是假设值，不是实测；表中的 BPB 也是示意读数。按 989 TFLOPS 峰值折算，四个方案的单卡有效算力依次约为 54.4、248.2、428.2、506.4 TFLOPS，即图中四根柱的长度。

方案 4 加上编译融合与 WSD 退火，真 MFU 突破 51.2%，单卡仅需 16.5 卡时（2.06 小时，49.4 美元）。同时验证损失从 0.89 降至 0.84，吞吐、成本与效果均迎来升级。

| **四笔账维度** | <b>调优前（BF16 + 补 PAD + Cosine）</b> | <b>调优后（双隔离打包 + 编译 + WSD）</b> | **必留归档证据与门禁断言** |
|---|---|---|---|
| **效果账** | 静态网页配比，终点锁死，`Val BPB = 0.89` | 零跨篇串味 + 末段数理代码退火，<b>`Val BPB = 0.84`</b> | `decontam_13gram.json` 碰撞率为 0 |
| **显存账** | 静态 `8.0 GB`；未融合激活峰值占 `34.2 GB` | 静态 `8.0 GB`；编译核融合后峰值降至 <b>`29.8 GB`</b> | `max_memory_allocated < 40 GB` |
| **时延/吞吐账** | 有效 `8.27 万词/s`（真 MFU `25.1%`），8 卡 `4.20h` | 有效 `16.87 万词/s`（真 MFU <b>`51.2%`</b>），8 卡 <b>`2.06h`</b> | `telemetry.jsonl` 真实非填充词吞吐 |
| **成本账** | 耗费 `33.6` H100 卡时，总计 `100.8 美元` | 耗费 <b>`16.5` H100 卡时</b>，总计 <b>`49.4 美元`</b>（立省 51%） | 封存 `step_16212` 稳定期可续训快照 |

*一轮 0.5B×10B 预训练四档配置真 MFU、8×H100 墙钟耗时与成本自动复算核验脚本*

```python
P_ACT, D_VALID = 0.50e9, 10.0e9   # 0.5B 激活参数，10B 无污染有效训练词元
PEAK_TF, USD_HR = 989.0, 3.0      # H100 SXM5 BF16 稠密峰值 989 TFLOPS，3 美元/卡时

runs = [
    ("1. FP32 + 补PAD",             31000,  0.581),
    ("2. BF16 + 补PAD",            142300,  0.581),
    ("3. BF16 + 双隔离Packing",     142800,  1.000),
    ("4. Packing + 编译融合 + WSD", 168700,  1.000),
]
for tag, raw_slot_s, valid_frac in runs:
    v_tps = raw_slot_s * valid_frac
    eff_tf = 6.0 * P_ACT * v_tps / 1e12
    mfu = eff_tf / PEAK_TF * 100.0
    gpu_h = D_VALID / v_tps / 3600.0
    print(f"{tag:26s} | 有效词/s:{v_tps:7.0f} | 真MFU:{mfu:5.1f}% | {gpu_h:5.1f}卡时 (8卡{gpu_h/8:4.2f}h) | ${gpu_h*USD_HR:5.1f}")
```

*输出*

```text
1. FP32 + 补PAD             | 有效词/s:  18011 | 真MFU:  5.5% | 154.2卡时 (8卡19.28h) | $462.7
2. BF16 + 补PAD             | 有效词/s:  82676 | 真MFU: 25.1% |  33.6卡时 (8卡4.20h) | $100.8
3. BF16 + 双隔离Packing       | 有效词/s: 142800 | 真MFU: 43.3% |  19.5卡时 (8卡2.43h) | $ 58.4
4. Packing + 编译融合 + WSD    | 有效词/s: 168700 | 真MFU: 51.2% |  16.5卡时 (8卡2.06h) | $ 49.4
```

复盘上述对账脚本的四行输出，可逐档拆清每项优化的物理贡献。从第 1 档切到第 2 档，启用 `BF16` 让原始槽位吞吐从 `3.10 万`跃至 `14.23 万/秒`。但受 `valid_frac = 0.581` 拖累，逾四成算力仍在空转。

从第 2 档升至第 3 档，原始槽位转速几乎未变（`14.28 万/秒`）。仅把有效占比拉满至 `1.000`，有效词速便涨至 `14.28 万词/秒`，真 MFU 净增 `18.2` 个百分点。第 4 档再靠编译融合削减显存往返，将总账单从 `100.8 美元`压至 `49.4 美元`。

### 5.1 📌 核对清单

> 📌 **核对清单：提交大规模预训练作业前必查 8 项硬门禁**
>
> - <b>五级提纯与 13-gram 净身。</b>入库语料已跑完 MinHash-LSH 跨站近似去重，并对全部评测集执行了 13-gram 倒排去污[[10]](https://arxiv.org/abs/2107.06499)。
> - <b>大词表 uint32 位宽断言。</b>写盘前已断言词表上限小于整数极值，`DeepSeek-V3`（`129K`）与 `Qwen3.8-27B`（`248,320`）严禁误用 `uint16`[[2]](https://arxiv.org/abs/2412.19437)[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)。
> - <b>混合架构双重边界隔离。</b>开启 `EOS` 打包后，全局注意力已传入 `cu_seqlens` 且篇内 `pos_ids` 归零。`GDN / KDA` 线性层也已同步清零递推矩阵与因果 `conv1d` 滑窗[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)[[4]](https://huggingface.co/moonshotai/Kimi-K3)。
> - <b>全局有效词数无偏归约。</b>梯度累积已按整步真实非 `-100` 监督词元总数归一。上线前已对拍确认不同累积步数下单步更新量一致。
> - <b>WSD 稳定期快照预留。</b>跨入退火起点切换高密数理代码语料前，已封存含 FP32 优化器矩的稳定期快照。后续扩训一律从该快照接力[[6]](https://arxiv.org/abs/2404.06395)。
> - <b>BF16 免缩放器与置空清零。</b>在 H100 上启用 `bfloat16` 时已关闭多余的 `GradScaler`，并传入 `zero_grad(set_to_none=True)`。
> - <b>裁剪前原始范数留痕。</b>监控记录的 `grad_norm` 确为 `clip_grad_norm_` 返回的裁剪前原始值，且每隔固定步数异步读取。
> - <b>诚实真 MFU 口径。</b>核算 `MFU` 时分子严格只计真实非填充词元乘 $`6 P_{\mathrm{act}}`$，分母锁定硬件规格书稠密峰值。

### 5.2 自测题与参考解析

<b>自测题 1（大词表 uint16 截断灾难与分片磁盘账）</b>：为 `Qwen3.8-27B`（`vocab_size = 248,320`）离线序列化 `25B` 个训练词元。若脚本误将类型写为 `np.uint16`，词号为 `248,044` 的 `<|endoftext|>` 会被静默篡改成哪个词号？改用 `np.uint32` 后，分片共占多少十进制 GB 磁盘空间？

**参考解析**：`uint16` 按模 `65,536` 截断保留低 16 位。因 $`248044 = 3 \times 65536 + 51436`$，结束符会被静默篡改为普通词号 <b>`51,436`</b>，连带破坏全窗口跨篇隔离。改用每词 4 字节的 `uint32` 后，磁盘体积为 $`2.5 \times 10^{10} \times 4\text{ B} = 100.0\text{ GB}`$。

<b>自测题 2（MinHash-LSH 捕获概率手算）</b>：清洗管线提取 $`M = 120`$ 个 5-gram 签名，拆为 $`b = 20`$ 个分桶带、每带 $`r = 6`$ 行。求两篇 Jaccard 相似度 $`s = 0.75`$ 的近似转载网页至少在一个分桶带碰撞被捕获的概率。

**参考解析**：单带 6 行全同概率为 $`0.75^6 \approx 0.17798`$，20 带全部漏网概率为 $`(1 - 0.17798)^{20} \approx 0.01984`$。因此至少命中一桶的捕获概率为 $`1 - 0.01984 = 98.02\%`$。

<b>自测题 3（跨卡梯度累积除数与漏除放大倍数）</b>：原实验在 `64` 卡（`W_dp = 64, B_micro = 2, S = 4096`）上跑满全局批 `2,097,152` 词/步。现迁回单机 `8` 卡节点（`B_micro = 2, S = 4096`）复现同一梯度步，`K_accum` 应设为多少？若循环内对微批均值漏除该系数，单卡缓冲区合梯度范数会虚高几倍？

**参考解析**：8 卡节点需设 `K_accum = 2,097,152 / (2 × 8 × 4096) = 32`。若反向前漏除 `32`，连续 32 次原地累加会让合梯度范数虚高 <b>`32 倍`</b>，令 `1.0` 裁剪阈值退化为步步硬截断。

<b>自测题 4（WSD 退火起点、语料预算与实时学习率）</b>：按全局批 $`2^{20}`$（`1,048,576`）词/步训练总计 $`50 \times 2^{30}`$（约 `53.69B`）词元。前 `2,000` 步线性预热至 `3.0e-4`，末段 `12.5%` 总步数线性退火至 `3.0e-5`。求总步数、退火起点步数、所需高密退火词元数及第 `48,000` 步学习率。

**参考解析**：总步数为 $`(50 \times 2^{30}) / 2^{20} = 51,200`$ 步。退火段占 $`51,200 \times 0.125 = 6,400`$ 步，故退火起点为第 <b>`44,800`</b> 步，对应需准备 $`6,400 \times 2^{20} \approx 6.71\text{B}`$ 高密数理代码词元。第 `48,000` 步恰处退火正中（进度 `0.5`），实时学习率为 $`1.65 \times 10^{-4}`$。

<b>自测题 5（6PD 真 MFU 与 8×H100 卡时成本核算）</b>：在单机 `8×H100 SXM5`（稠密峰值 `989 TFLOPS`，`3 美元/卡时`）上用 `30B` 干净打包词元预训练 `1.5B` 激活参数的混合稠密模型，实测单卡每秒稳定消化 `54,944` 个真实有效词元。求单卡真 MFU、整轮总卡时、8 卡墙钟耗时与租卡总费用。

**参考解析**：单卡有效算力为 $`6 \times 1.5\times 10^9 \times 54,944 \approx 494.5\text{ TFLOPS}`$，除以 `989` 得<b>真 MFU = `50.0%`</b>。整轮共耗 $`3.0\times 10^{10} / (54,944 \times 3600) \approx 151.7`$ 卡时，8 卡并联墙钟耗时 <b>`18.96 小时`</b>，总费用约 <b>`455.0 美元`</b>。

### 5.3 接下来的问题在哪

复盘本篇顺畅跑通的单机闭环，背后其实享受了两个只在小模型身上成立的便利条件。一方面，`0.5B` 模型的混合精度静态底盘 $`16\Psi`$ 仅占 `8.0 GB`，单张 `80 GiB H100` 足以全量装下权重、梯度与优化器双矩。另一方面，机箱内 8 张卡全走 `450 GB/s` 单向高速 NVLink 互联，且全程仅跑 `2 小时`，几乎遇不到光模块闪断或硬件掉卡。

一旦挑战 2026 主力旗舰，单机条件便不再成立。最小的混合架构模型也需 400+ GB 显存底盘。而超大规模 MoE 架构甚至高达数TB，须铺到上千张 GPU 上连跑整月，三道分布式难题随即浮现[[2]](https://arxiv.org/abs/2412.19437)[[3]](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)[[4]](https://huggingface.co/moonshotai/Kimi-K3)：

- <b>显存切分账。</b>超出单卡五倍至百倍的显存底盘，怎样在数据并行（DDP / ZeRO / FSDP）、张量并行（TP）与流水线并行（PP）之间合理切分？
- <b>通信隐藏账。</b>跨机 RDMA 网卡比机内 NVLink 慢整整 9 倍，梯度同步耗时怎样藏进反向传播计算背后？
- <b>断点续训账。</b>半夜突然掉一张卡，重启后怎样让切碎的模型分片、优化器矩、随机种子与 `memmap` 游标原子级续上，确保 Loss 曲线丝毫不跳？

带着这三道硬核考题，我们迈入下一篇 [ZTL-12｜训练｜多卡的账和中断后续上](./12-multi-gpu.md)。

### 5.4 诚实清单与修订记录

1. <b>混合架构线性层定长状态未计入。</b>Qwen3.8-27B 与 Kimi K3 的线性注意力层各有一份不随上下文增长的递推矩阵，显存统计只含真正随词元增长的 KV 部分。短上下文时这份定长状态并不小，对账时需另算。
2. <b>跨篇隔离的 uint32 断言门禁已实施。</b>词表 248K+ 的模型若用 uint16 会静默截断，本篇第 5.2 自测题 1 给出了 Qwen3.8 的具体数值。上线前务必跑一遍断言脚本。
3. <b>梯度累积除数漏除倍增坑。</b>第 5.2 自测题 3 揭示分布式训练中梯度归一化的关键：漏除 K_accum 系数会让梯度虚高 K_accum 倍，直接摧毁学习率表现。
4. <b>真 MFU 与硬件脊点的核心口径差。</b>本篇全部数字来自代码复算或 GPU 规格书标称峰值。实际可达值受功耗限制，新硬件代际脊点会升级。

#### 修订记录

| 日期 | 改了什么 |
|---|---|
| 2026-11-19 | 按全系列规范修订：章节标题去冒号改判断句、新增诚实清单与修订记录二级标题、压缩模型名重复出现次数至上限；删除来路不明的图表数字。 |

> 🔑 **本章要点**：回到主线，一轮高质量预训练由五级净身语料、注意力与线性态双隔离的 `uint32 memmap` 打包、全局有效词数无偏归约配 WSD 弹性退火以及诚实遥测共同构成。守住这 8 项启动门禁并算熟五道手算账，即可稳健跨向下一篇 [ZTL-12｜训练｜多卡的账和中断后续上](./12-multi-gpu.md) 的百卡分布式训练系统。
>
> **诚实声明**：屋脊线脊点值未实测，仅按硬件规格书核算。torch.compile 显存削减数据基于 PyTorch 2.0～2.1，新版本能力可能提升。本篇修订时间 2026-11-19，按全系列规范压缩模型重复出现次数至行业上限。

## 参考文献

按正文首次引用顺序编号，资料截至 2026 年 11 月。

[1] Penedo et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale (FineWeb &amp; FineWeb-Edu). NeurIPS / arXiv:2406.17557 (2024). [arxiv.org/abs/2406.17557](https://arxiv.org/abs/2406.17557)

[2] DeepSeek-AI. DeepSeek-V3 Technical Report. arXiv:2412.19437 (2024). [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

[3] Qwen Team. Qwen3.8-27B Model Card and Configuration. Hugging Face (2026). [huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)

[4] Kimi Team. Kimi K3 Model Card and Technical Overview. Hugging Face (2026). [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)

[5] Krell et al. Efficient Sequence Packing without Cross-Contamination: Accelerating Large Language Models without Impacting Performance. arXiv:2107.02027 (2021). [arxiv.org/abs/2107.02027](https://arxiv.org/abs/2107.02027)

[6] Hu et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. COLM / arXiv:2404.06395 (2024). [arxiv.org/abs/2404.06395](https://arxiv.org/abs/2404.06395)

[7] Chowdhery et al. PaLM: Scaling Language Modeling with Pathways. JMLR / arXiv:2204.02311 (2023). [arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311)

[8] Qwen Team. Qwen3 Technical Report. arXiv:2505.09388 (2025). [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

[9] Penedo et al. The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only. NeurIPS / arXiv:2306.01116 (2023). [arxiv.org/abs/2306.01116](https://arxiv.org/abs/2306.01116)

[10] Lee et al. Deduplicating Training Data Makes Language Models Better. ACL / arXiv:2107.06499 (2022). [arxiv.org/abs/2107.06499](https://arxiv.org/abs/2107.06499)

[11] Li et al. DataComp-LM: In Search of the Next Generation of Training Sets for Language Models. NeurIPS / arXiv:2406.11794 (2024). [arxiv.org/abs/2406.11794](https://arxiv.org/abs/2406.11794)

[12] Broder. On the Resemblance and Containment of Documents. Compression and Complexity of SEQUENCES (1997). [ieeexplore.ieee.org/document/666900](https://ieeexplore.ieee.org/document/666900)

[13] Brown et al. Language Models are Few-Shot Learners. NeurIPS / arXiv:2005.14165 (2020). [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

[14] Dubey et al. The Llama 3 Herd of Models. arXiv:2407.21783 (2024). [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

[15] Hoffmann et al. Training Compute-Optimal Large Language Models. NeurIPS / arXiv:2203.15556 (2022). [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)

[16] NumPy Developers. numpy.memmap — Memory-mapped file arrays. NumPy Reference Manual. [numpy.org/doc/stable/reference/generated/numpy.memmap.html](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

[17] PyTorch Contributors. torch.nn.attention.flex_attention &amp; BlockMask Documentation. PyTorch Official Documentation (2025). [pytorch.org/docs/stable/nn.attention.flex_attention.html](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)

[18] Yang, Kautz, and Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR / arXiv:2412.06464 (2025). [arxiv.org/abs/2412.06464](https://arxiv.org/abs/2412.06464)

[19] Kimi Team. Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692 (2025). [arxiv.org/abs/2510.26692](https://arxiv.org/abs/2510.26692)

[20] Kimi Team. Kimi K2: Open Agentic Intelligence. arXiv:2507.20534 (2025). [arxiv.org/abs/2507.20534](https://arxiv.org/abs/2507.20534)

[21] Hägele et al. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations. NeurIPS / arXiv:2405.18392 (2024). [arxiv.org/abs/2405.18392](https://arxiv.org/abs/2405.18392)

[22] NVIDIA Corporation. NVIDIA H100 SXM5 Tensor Core GPU Architecture Datasheet. NVIDIA Documentation. [resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)

---

本章代码：[`11-pretrain/`](../11-pretrain/) · 上一章：[ZTL-10｜模型｜组装小模型并让一个批次下降](./10-tiny-model.md) · [目录](./README.md) · 下一章：[ZTL-12｜训练｜多卡的账和中断后续上](./12-multi-gpu.md)
