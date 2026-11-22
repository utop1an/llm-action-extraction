# RISeC / RiCoRecA 的 cEASDRL 训练复核

复核日期：2026-11-22（Australia/Sydney）。本次重新运行数据、模型、经验回放、无标注推理和评分检查，没有重新训练，也没有改写历史数据、权重或结果。

## 结论

两套实验确实执行了从头训练，保存的模型能够重新产生原来的测试预测，动作—参数对应关系参与最终评分。但准确名称应是 **cEASDRL CNN/DQN 的 BIO 完整片段抽取适配版**，不能称为“原版 cEASDRL 不改方法的严格复现”。

- 用户要求的不区分 ES/EX/OP、从文本预测动作名称及对应参数名称、端到端 F1：两套实验均已实现。
- RiCoRecA：执行了用户指定的文档级 64/16/20 固定划分；训练只读取 train/dev，选好两个模型后才做测试。
- RISeC：保留官方可用文档的 126/56/78 划分，并非 64/16/20；保留了 45 条未明确关系标签 TODO，且历史候选训练会自动输出测试分数。其结果可以重算，但不是全程封存测试集的干净确认性实验。
- 当前两套模型都用了 BIO、平衡经验采样、mention 数量奖励塑形、独立目标网络以及改变后的更新调度。这些是方法变更，不能只说“换了数据集和运行框架”。

最终端到端测试结果：

| 数据集 | 动作名称 P | 动作名称 R | 动作名称 F1 | 关联参数 P | 关联参数 R | 关联参数 F1 |
|---|---:|---:|---:|---:|---:|---:|
| RISeC，现有标注口径 | 95.57% | 97.95% | **96.75%** | 75.33% | 89.54% | **81.82%** |
| RiCoRecA，ACTION 及显式成员投影 | 95.40% | 98.35% | **96.85%** | 52.09% | 75.96% | **61.80%** |

两者语料、标注范围、划分比例和训练流程的测试封存程度不同。不能据此认定某一数据集更优，也不能直接与原论文表中的分数比较。

## 1. 原版究竟是什么；10 折的纠正

核对的论文是 *NLtoPDDL: One-Shot Learning of PDDL Models from Natural Language Process Manuals*，2020，Miglani 与 Yorke-Smith。

论文第 4 节明确写明：**cEASDRL 使用 64/16/20，不使用交叉验证**。论文比较的早期 **EASDRL** 报告的是 **10 折**。因此，“cEASDRL 原实验是 10 折”的说法不正确。

本项目对照的作者代码提交：

- 仓库：<https://github.com/Shivam-Miglani/contextual_drl>
- 固定提交：`b6b2acb8e5cc8009adb2dacfb363e0157a08f922`
- 论文：<https://icaps20subpages.icaps-conference.org/wp-content/uploads/2020/10/KEPS-2020_paper_16.pdf>
- 架构：<https://github.com/Shivam-Miglani/contextual_drl/blob/b6b2acb8e5cc8009adb2dacfb363e0157a08f922/EADQN.py>
- 训练入口：<https://github.com/Shivam-Miglani/contextual_drl/blob/b6b2acb8e5cc8009adb2dacfb363e0157a08f922/cmain.py>

原方法用两个 DQN：先选动作词，再针对动作选参数词。词表示可以使用不同预训练编码器。本项目选用了其 GloVe + ELMo 变体，并没有复现所有 BERT/Flair 变体，也没有重做原三数据集论文表格。论文后续 PDDL/iLOCM 阶段不属于本次抽取实验。

RiCoRecA 也不能简单说“没有官方划分”：作者发布了 Cross_Val 文件，代码使用五折，再把每个留出块分成验证和测试。当前实验只用这些文件的并集确定发布语料范围，**重新做用户指定的 64/16/20**，没有沿用其折划分。

依据：<https://github.com/FilipposVentirozos/RiCoRecA/blob/af80d0c3fdaa5a375625bb9f006ce536770068f2/recipe-transformation-evaluation/evaluation.py>，`generate_cross_val_predict_files`。作者代码另有非交叉验证生成函数，不能把所有发布协议混为一个。

## 2. 数据如何进入实验

### RISeC

来源：`YiweiJiang2015/RISeC`，提交 `70b9a0b4c4eb656375a83bc1c0ca47c29ec2da2c`。原始 BRAT 文件位于 `data/procedural/risec/upstream/data/`。

保留原 train/dev/test 目录。按 `.txt` 配对 `.ann` 后，共 260 篇。存在一份没有对应文本的 `train/165_Coq_au_Vin_alla_Italiana.ann`，不能构成训练样本，已登记。

| 划分 | 文档 | 动作 | 当前口径的动作—参数关系 | 其中 TODO |
|---|---:|---:|---:|---:|
| train | 126 | 1,487 | 2,692 | 29 |
| dev | 56 | 694 | 1,212 | 6 |
| test | 78 | 880 | 1,616 | 10 |
| 合计 | 260 | 3,061 | 5,520 | 45 |

126/56/78 约为 48.46%/21.54%/30%，不是 64/16/20，也不是五折或十折。

动作目标来自 `AC` 实体。参数目标保留从动作指向显式文本片段的关系，排除 `_X`、动作指向动作、条件从句等既定范围之外的标注；不预测语义角色。名称和位置从原文标注恢复，不用首词代替完整名称。

**问题：旧转换规则没有把 TODO 过滤掉。** 这 45 条未明确关系进入了训练或评估，不能将当前标注口径描述成“全部经过定义的参数关系”。当前分数保留原口径以便可复查，没有偷偷修改 gold 来替换历史结果。

### RiCoRecA

来源为用户提供的 `data/data.zip`。压缩包 SHA256：

`921f22602f1af88d21bb40038204f9e0bf52214c866ac5a889a277901f12ba2c`

主标注为 `Annotator_P`。原始 JSON 有 163 篇；已发布 Cross_Val CSV 并集包含 156 篇。本实验使用全部 156 篇发布语料，没有在这个范围中丢弃困难样本。其余 7 篇的 ID 已记录，其中 2 篇原始 answer 为 ignore；不能据此推断其余 5 篇被作者排除的具体原因。

注意统计口径：全部 raw 有 2,290 个 ACTION；属于 156 篇发布语料的 raw 有 2,177 个；发布版目标为 **2,176 个**。发布版比对应 raw 少了 1 个 ACTION 和 5 个工具/工具部分实体片段，差异见 `data/procedural/ricoreca/release_differences.json`。

| 划分 | 文档 | 动作 | 动作—参数关系 |
|---|---:|---:|---:|
| train | 99 | 1,313 | 2,616 |
| dev | 25 | 378 | 738 |
| test | 32 | 485 | 936 |
| 合计 | 156 | 2,176 | 4,290 |

划分算法在 `scripts/prepare_ricoreca_split.py`：

```python
train_dev, test = train_test_split(sorted(document_ids), test_size=0.2, random_state=1)
train, dev = train_test_split(train_dev, test_size=0.2, random_state=1)
```

先预留 20% test，再从剩余 80% 中预留 20% dev。整数取整后为 99/25/32，实际占比 63.46%/16.03%/20.51%。**划分 seed=1，训练 seed=0**，二者用途不同。

先按整篇食谱划分，再展开其动作及参数训练样本。同一篇中的所有动作只属于同一个集合。检查未发现重复文档 ID 或跨集合的规范化全文完全重复；这不等同于完成了语义近重复检索。

参数关联的转换并非最近动词匹配：结合发布 CSV 的 membership 名称和原始 Prodigy 的位置关系，沿 Member/Or/Modifier 查找并在遇到谓词时停止，再处理 Join 共享；不把 Dependency 当成参数连接。重复动词必须定位到唯一动作实例，否则转换报错。

最终任务只保留 **ACTION 及其显式文本成员**：包括食材、工具、指代词、部分实体、状态、度量、设置、目的等。不预测这些类型，只预测名字和对应动作。IF/UNTIL/REPEAT 不作为动作，它们独占的 662 个成员 mention 不在本任务中。显式指代词仍按原文抽取，不补成隐含实体或推断的先行词。

有 1 条 raw 关系指向未标注词，已忽略并登记；剩余任务目标均能解析。4,290 条保留关系均唯一定位，原文/位置对应、BIO 往返与窗口覆盖均通过。**“无损”仅指这个已声明的任务投影，不指完整 RiCoRecA 标注体系。**

## 3. 所谓动作模型、参数模型具体做什么

它们都是文本抽取器，不是 PDDL 动作 schema 学习器。

例如任务定义中的 `Chop the red onion.`：第一阶段输出 `Chop` 的文本位置；第二阶段以该位置为条件输出 `red onion` 的完整文本位置。这个句子用于说明任务，不是假称某条实测预测。

**动作阶段**：一篇文档一个样本，逐词决定 O/B/I，恢复完整动作片段。

**参数阶段**：一个动作一个样本，以该动作首词为中心，确定最多 100 token 的固定窗口；再逐词输出属于这个动作的参数片段。同一段文字可以针对不同动作重复输入，因为参数归属不同。距离特征为词到动作首词的绝对距离。

O 表示非目标，B 表示名称开始，I 表示名称继续。BIO 只是名称边界编码，**与 ES/EX/OP 没有关系**。解码时，O 后单独出现的 I 按新片段开始处理，不使用 gold 修正边界。

训练参数阶段需要用训练集的 gold 动作构造样本；验证参数阶段也使用 dev 的 gold 动作来选该阶段 checkpoint。最终测试则用第一阶段的**预测动作**，不能将 gold 动作位置提供给参数模型。主结果使用完整的两阶段链路。

本项目最长文档：RISeC 545 tokens，RiCoRecA 400 tokens。动作输入容量 600，参数窗口容量 100；在各自声明的 gold 目标范围内没有窗口外参数被静默丢弃，也没有 BIO 边界损失。

## 4. 实际模型与训练过程

### 编码器与 CNN

- 冻结 GloVe 100 维 + ELMo-small 768 维，总共 868 维。词向量由旧版 Flair 环境生成并缓存；不在这些训练集上更新 ELMo。
- 动作阶段输入：词向量和已经作出的序列决策。参数阶段再增加到指定动作的距离特征。
- 卷积宽度 2/3/4/5，每种 32 个过滤器，ReLU 后全局最大池化。
- 拼接得到 128 维，接 256 维 ReLU 全连接层，再接 3 个 Q 值。
- 正式训练使用 PyTorch 实现，未直接运行原作者完整 Keras 训练循环。旧 Keras 环境用于词向量及数值兼容检查。
- 两个抽取网络都随机初始化、从头训练；RiCoRecA 没有加载 RISeC 的抽取模型权重。复用的是实现、配置和固定的预训练词表示。

`third_party/contextual_drl/UPSTREAM.md` 中关于直接使用原训练组件的描述针对早期 `ceasdrl_local.py` 兼容实验；不能据此推断本报告中的正式 BIO 实验使用了原训练循环。

### DQN 更新

每条文本作为一个 episode，按顺序对当前词作 O/B/I 决策。状态中的未处理位置为 0，处理后记录所选决策；训练奖励查询该位置的训练标签。

正确 O 基础奖励 +50，错误 O 标签决策按 gold 类别给负奖励；gold 为 B/I 时，精确预测对其类别才获得 +100，预测错为 -100。另加与已预测 mention 数量相关的 `250*r^2` 奖励塑形，以及终止时 +2。这里 `r` 的分母是容量 600 或 100，而非正文实际长度。

Bellman 目标为 `reward + 0.9 * max Q_target(next_state)`，终止时只取 reward。使用 MSE 和 Adam 更新网络。它不是 BIO 交叉熵分类器，也不是训练 LLM。

| 配置 | 当前两套正式实验 |
|---|---|
| 最大 epoch | 每阶段 10 |
| Early stopping | dev 连续 3 轮无改进 |
| 学习率 | 0.001 |
| Adam epsilon | 1e-8 |
| gamma | 0.9 |
| replay 容量 | 50,000 |
| 优化 batch | 32 |
| 并行 episode | 16 |
| 优化频率 | 每 4 个向量化步一次 |
| target | 独立参数，每 100 次优化更新同步 |
| exploration | epsilon 从 1 降至 0.1，约第 1 个 epoch 的真实 token 总量内完成；测试为贪心 |
| replay | 正奖励中平衡正确 O/B/I；batch32 的正奖励配额 int(0.9*32)=28，即 87.5% |
| checkpoint | 各阶段单独按 dev 完整片段 micro F1 取最高；并列保留更早 epoch |

训练实际完成情况：

| 数据集/阶段 | 训练样本 | 完成 epoch | 最佳 epoch | 最佳 dev F1 | 环境 token 步数 | 优化更新次数 | 设备 |
|---|---:|---:|---:|---:|---:|---:|---|
| RISeC 动作 | 126 | 10 | 8 | 95.88% | 152,850 | 5,600 | RTX 4070 |
| RISeC 参数 | 1,487 | 10 | 10 | 80.67% | 1,391,140 | 23,250 | RTX 4070 |
| RiCoRecA 动作 | 99 | 10 | 10 | 94.76% | 123,410 | 4,668 | CPU |
| RiCoRecA 参数 | 1,313 | 10 | 9 | 60.80% | 1,241,000 | 20,745 | RTX 4070 |

表内参数 dev F1 是 gold-action-conditioned 指标，用于选参数 checkpoint，不是端到端验证 F1。RiCoRecA 动作阶段因本机 CUDA 持续吞吐异常改为 CPU；模型和训练目标没有因此改变，但不能承诺换设备后重新训练得到逐位相同结果。正式运行没有载入那些中止的吞吐诊断权重。

每阶段完成 10 轮是训练预算事实，不证明已充分收敛。两套仅报告单个正式训练 seed 的结果，没有多 seed 均值或标准差。

## 5. 与原版 cEASDRL 的区别

| 项目 | 原作者论文/代码 | 当前正式实验 | 性质 |
|---|---|---|---|
| 两阶段抽取 | 先动作、后条件参数 | 保留 | 核心结构一致 |
| 词表示 | 提供 GloVe+ELMo 等变体 | GloVe100+ELMo768 | 选定其中一种 |
| CNN 主干 | 2/3/4/5 卷积、32 filters、256 dense | 保留；重复特征通道以代数等价方式计算 | 数值验证的移植 |
| 输出决策 | select/reject，2 个 Q | O/B/I，3 个 Q | 方法改变 |
| 经验采样 | 按奖励正负偏采样 | 正奖励内再平衡正确 O/B/I | 方法改变 |
| 奖励塑形计数 | 已选 token | 已解码 mention 开始位置 | 方法改变 |
| target 参数 | 固定提交用相同 layer tensors 构造两个 Model，实际共享权重 | deepcopy 后独立同步 | 行为修正，但不再原样复现 |
| 优化/同步频率 | `cmain.py` 默认每 1 环境步训练，5 步同步 | 16 个 episode 向量化，每 4 向量步优化、100 优化步同步 | 训练动态改变 |
| epsilon 衰减 | 代码默认 1,000 步 | 按首 epoch 训练 token 总量 | 训练动态改变 |
| 动作容量 | 500 | 600 | 避免当前 RISeC 长文档截断 |
| 参数上下文 | 前句+当前句等作者构造逻辑，容量100 | 动作位置确定的固定100-token窗口 | 输入上下文改变 |
| epoch/选模 | 论文主要报告1epoch，代码含训练中验证 | 最多10epoch，每epoch验证，patience3 | 实验预算与选择时点改变 |
| 数据/评分 | 原三语料及其抽取计分 | 新数据投影，完整片段及动作归属严格匹配 | 任务与评价改变 |

上表对代码参数明确指默认值；并不把默认值自动视为每次作者实验的完整实际配置。

原 Keras 的关键事实是 `self.model = Model(inputs, out)` 和 `self.target_model = Model(inputs, out)` 使用同一组层。独立目标网络更符合通常 DQN 实现，但本身就是与该提交的差异，不能因称其为修复就忽略。

保存的兼容性检查：二分类核心 Q 值最大误差约 1.07e-4，相对 loss 误差约 1.06e-7，梯度相对 L2 误差不超过 1.28e-6。训练后 BIO 权重也装回作者 Keras 主干，分别改为三输出后前向比较：RISeC 动作/参数最大 Q 误差为 3.05e-4/1.22e-4，RiCoRecA 两阶段均约 1.83e-4。

这些是历史数值兼容检查，本次核对其记录和对应工件，并重新运行了 PyTorch 全测试集推理；**本次没有重跑 Keras 训练或重做完整训练轨迹**。前向相容不能证明 BIO 任务、采样、奖励和训练过程与原论文等价。

## 6. 最终 F1 究竟怎么计算

模型输入测试文本，先产生动作片段，再对每个预测动作生成参数窗口与参数片段。解码阶段不读取 gold 动作、参数、边界、名称或类型。

动作匹配键为 `(文档ID, 动作开始位置, 动作结束位置)`，参数匹配键为 `(文档ID, 动作开始位置, 动作结束位置, 参数开始位置, 参数结束位置)`。每个名称还验证必须是原文相应位置的完整子串。

因此：

- 只命中 `red onion` 中的 `onion` 不算完整名称正确。
- 识别了正确参数词但挂到错误动作上，不算关联参数正确。
- 同篇中两次出现的同名动作是两个不同实例。
- 漏掉动作后，其 gold 参数仍留在分母中成为 FN。
- 不以同义词、词形还原或部分重叠作为正确；不把非目标 O 标签纳入名称 F1。
- 先汇总所有测试文档的计数再算 micro F1，不平均每篇 F1，也不只评价成功预测出的动作。

`P=TP/预测数`，`R=TP/gold数`，`F1=2TP/(预测数+gold数)`。

| 数据集/指标 | TP | 预测数 | gold数 | FP | FN | F1 |
|---|---:|---:|---:|---:|---:|---:|
| RISeC 动作 | 862 | 902 | 880 | 40 | 18 | 96.75% |
| RISeC 关联参数 | 1,447 | 1,921 | 1,616 | 474 | 169 | 81.82% |
| RiCoRecA 动作 | 477 | 500 | 485 | 23 | 8 | 96.85% |
| RiCoRecA 关联参数 | 711 | 1,365 | 936 | 654 | 225 | 61.80% |

RISeC 参数训练 receipt 中另有 83.57% 的 test F1，那是给定 gold 动作的位置后评参数，仅为诊断值，**不能作为用户要求的端到端参数成绩**。主值是 81.82%。

早期 88.85% / 26.44% 对应旧首词模型在完整名称评分器下的复算，旧参数模型只有3轮训练。它们不是当前最终模型的成绩，也不能把它与现结果的差值解释为某一单独改动的净收益。

RiCoRecA 当前参数误报 654 个，P 为 52.09%，说明参数多抽仍是明显问题。该结果不是完整计划结构、参数角色、前提效果、共指消解或执行正确率。

## 7. 测试集协议和 RISeC 的待修复点

**RISeC：不能称全程封存测试。** `scripts/train_ceasdrl_risec.py` 会在每次候选训练结束后自动计算 test。保存的 `selection.json` 按 action dev F1 选择，且没有证据表明 test 样本参与梯度更新；但 test 分数已经可见，因此不能排除历史人工设计决策受到它影响。按 dev 排名不等于从未接触 test。

RISeC 原始转换也保留了 TODO。对同一份冻结预测，仅将 test 的10条 TODO 从参考集中去掉：关联参数 gold 1,606、TP 1,441、预测1,921，F1 **81.71%**。这是标注口径敏感性检查，不能宣称“清洗重训后81.71%”：训练仍用过29条 TODO，验证仍用过6条。

如果把已定义的条件从句关系也纳入参考，则同一预测的参数 F1 为78.72%。这进一步说明必须在论文中明确参数目标范围，不能只写笼统的“所有参数”。

**RiCoRecA：现有封存证据完整。** 训练 manifest 只有 train/dev 的数据哈希和样本，test 为空。两阶段 receipt 的 test 为 null 且 `test_evaluated=false`；freeze.json 先固定两组 checkpoint、数据、训练 receipt 和评价代码哈希，之后才运行测试。参数阶段的测试上下文由预测动作构造。本次核对与冻结文件相符。

但当前 test 现在已经评估并公开：未来基于这些测试错误修改方法，不能把这32篇重新称为全新未见测试集。用于后续正式确认时应预先固定流程，并如实说明测试使用历史。

## 8. 本次重新执行的验证

本次新增证据位于此目录；没有改写正式实验结果。

1. RISeC 523 个来源文件的 Git blob 哈希与本地内容相符；重新从 BRAT 字符位置复核标注、tokenization、关系与分组。
2. RiCoRecA 从 raw + 发布 membership 重新转换全部156篇，结果与保存JSONL一致；用 seed1 重建99/25/32的全部ID列表，一致。
3. 四个模型各检查 last.pt 中保留的50,000条经验，共200,000条。均来自 train，决策前后状态、gold BIO、终止标记与奖励一致。**这是当前缓存的全部经验，不是声称回放了历史所有百万级训练步骤。**
4. 检查数据、词表示缓存、训练源码、最佳权重与冻结清单哈希；检查各epoch完成记录、dev最佳epoch以及独立target参数。
5. 重载最佳模型，去掉推理输入的标注字段，以 batch7 重新预测全部110篇测试文本，与原 batch32 结果完全一致。
6. 以字符位置独立评分，所有TP、预测数、gold数及P/R/F1与正式报告一致。
7. 六个相关测试文件再次运行，**32 passed**。环境发出一条现有 SciPy/NumPy 版本兼容警告，但划分重建和测试通过；本次没有改动环境版本。

证据：`risec/audit.json`、`ricoreca/audit.json`、各自 `fresh_predictions.json`、`verification.json`。

## 9. 可复跑入口与后续实验边界

只复核、不重训的入口（输出目录必须不存在）：

```powershell
python scripts/audit_ceasdrl_risec.py --output results/my_new_review/risec
python scripts/review_ceasdrl_ricoreca.py --output results/my_new_review/ricoreca
```

当前 RiCoRecA 正式训练对应的参数如下。若确需重新训练，必须改用新的输出目录，保留原模型；以下是复现说明，本次没有执行训练：

```powershell
python scripts/train_ceasdrl_holdout.py --features data/procedural/ricoreca/features_train_dev --output results/my_new_ricoreca/act --parity-report results/ceasdrl_risec_20261118/parity/parity_report.json --mode act --device cpu --epochs 10 --patience 3 --seed 0 --env-batch 16 --batch-size 32 --train-frequency 4 --target-updates 100 --learning-rate 0.001 --span-labels --span-replay balanced --span-rate mentions
python scripts/train_ceasdrl_holdout.py --features data/procedural/ricoreca/features_train_dev --output results/my_new_ricoreca/arg --parity-report results/ceasdrl_risec_20261118/parity/parity_report.json --mode arg --device cuda --epochs 10 --patience 3 --seed 0 --env-batch 16 --batch-size 32 --train-frequency 4 --target-updates 100 --learning-rate 0.001 --span-labels --span-replay balanced --span-rate mentions
```

训练入口 `train_ceasdrl_holdout.py`、最终端到端评价入口 `evaluate_ceasdrl_holdout.py`；RISeC历史入口为 `train_ceasdrl_risec.py`、`evaluate_ceasdrl_spans.py`。生成词向量需要已经安装的旧版Flair环境；新PyTorch环境不直接替代该编码器版本。

如果要把论文中的方法名称写成“严格复现 cEASDRL”，尚需另外建立原二决策对照、对齐上下文/奖励/采样/更新频率与预算，明确原共享target行为和独立target修正版的区别，并先在原任务上验证。不能把当前BIO结果改个名字就当作严格复现。

如果目标是用户要求的完整动作—参数名称抽取，则当前适配方向可以保留；需先明确报告为 adapted cEASDRL。RISeC若作为正式比较基线，建议版本化清理TODO规则、采用只加载train/dev的训练流程，并保留原官方划分或另行明确重划分。原测试集已被使用，重训并不能消除这一历史事实。所有比较方法必须共享同一数据版本、同一划分和同一端到端评分器。
