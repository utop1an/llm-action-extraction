# RiCoRecA：cEASDRL 64/16/20 实验报告

本次已从头训练并完成封闭测试，独立复核通过。方法是当前项目的 **cEASDRL CNN/DQN BIO 适配版**，
不是原论文全部任务和分数的严格复现。不预测 ES/EX/OP，也不学习 PDDL 前置条件或效果。

## 测试结果

| 指标 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 完整动作名称 | 95.40% | 98.35% | **96.85%** | 477 | 23 | 8 |
| 对应的完整参数名称 | 52.09% | 75.96% | **61.80%** | 711 | 654 | 225 |

测试输入仅为文本。先预测动作，再针对每个预测动作提取参数。
动作和参数均要求完整名称及出现位置精确匹配；参数还必须属于正确的动作实例。
漏掉动作会使它的所有 gold 参数计为 FN。以上是 micro F1，不是 token F1，也不是喂入 gold 动作后的参数成绩。
token/name 评分器与独立字符位置评分器给出的 TP、预测数、gold 数和 F1 完全一致。

## 数据及固定划分

| 集合 | 文档数 | 动作数 | 动作—参数关联数 |
|---|---:|---:|---:|
| train | 99 | 1,313 | 2,616 |
| dev | 25 | 378 | 738 |
| test | 32 | 485 | 936 |
| 合计 | 156 | 2,176 | 4,290 |

排序文档 ID，用 seed=1 留出 20% test，再对剩余文档留出 20% dev。
名义比例 64/16/20，整数取整后约 63.46/16.03/20.51。不是五折或十折。
同一食谱不会跨集合；重复 ID 和规范化全文重复均已检查。

原始 Annotator_P 有 163 篇，作者发布的 Cross_Val CSV 并集为 156 篇。
采用这 156 篇的发布版本重新划分，7 篇未发布原始记录的 ID 完整保留在 split manifest。
原始全部 ACTION 计数 2,290，不等于这次实际训练的 2,176。
发布范围内有 6 个 raw span 未进入发布标注，其中包括 1 个 ACTION `absorbed`；沿用发布版本，
没有自行把这些 span 加回目标。差异记录见 `data/procedural/ricoreca/release_differences.json`。

参数来源包括食材、工具、指代/部分实体，以及与 ACTION 关联的状态、计量、设置、目的片段。
IF/UNTIL/REPEAT 控制谓词及仅属于它们的 662 个 member mention 不计入本任务；不填补隐式对象。
每条关联通过发布 membership 与原始位置关系核对，重复动词被定位到具体出现位置。
不采用最近动词猜测，Dependency 不当作参数边。发现 1 条失效原始关系并记录；保留目标解析完整。
所有 BIO 边界可无损还原，600/100-token 输入容量覆盖全部保留目标，无截断或静默丢弃。

## 训练和选模

冻结 GloVe-100 + ELMo-small-768；两个阶段权重均从头初始化，不使用 RISeC 训练后的权重。
训练 seed=0，最多 10 epochs，patience=3，Adam lr=0.001，balanced BIO replay、mention-rate shaping。
每阶段只按 dev exact-span micro F1 选择 checkpoint。BIO 和 balanced replay 是对原方法的明确适配。

| 阶段 | 设备 | 完成轮数 | 选中轮次 | 验证 F1 | 耗时 |
|---|---|---:|---:|---:|---:|
| 动作阶段 | cpu | 10 | 10 | 94.76% | 41.6 分钟 |
| 参数阶段 | cuda | 10 | 9 | 60.80% | 46.2 分钟 |

表中的参数阶段验证 F1 使用正确动作作为条件，只用于选择该阶段的 checkpoint，不能替代上面的端到端测试 F1。

600-token 动作训练的 CUDA 运行出现持续异常变慢，正式动作训练改用同一 PyTorch 实现的 CPU。
中断的 GPU 诊断未完成一个 epoch，不计入结果。实际设备以表格及配置文件为准。
每轮完成全部训练样本，检查梯度有限、权重确实更新和保存后重新载入的一致性。
已将两个最佳模型的权重加载到原始 Keras DeepQLearner；每阶段检查 4 个验证上下文，
前向 Q 值最大绝对误差为 0.00018311，数值一致性检查通过。
数据转换、边界、关联计分和测试隔离等 32 项回归检查通过。
训练特征只含 train/dev，test 为空；两阶段模型哈希冻结后，才读取测试文本进行解码和评分。
单次固定划分、单个训练种子，没有交叉验证均值或多种子标准差。

## 与现有 RISeC 记录并列

| 数据集与划分 | 动作名称 F1 | 对应参数名称 F1 |
|---|---:|---:|
| RISeC，官方 126/56/78 | 96.75% | 81.82% |
| RiCoRecA，本次 99/25/32 | 96.85% | 61.80% |

这里只是不同数据集结果的并列记录。语料、标注范围、划分和运行设备不完全相同，
不能用两行分数判断数据集质量，也不能当作同一测试集上的方法改进。
RISeC 来源：`results/ceasdrl_risec_20261118/spans_v2/pipeline_seed0/receipt.json`；此次没有重训或覆盖旧模型。

## 结果文件和复现入口

- `final/receipt.json`：指标、协议、设备、模型和数据哈希。
- `final/predictions.json`：每篇文本的动作、参数名称和 token/字符位置。
- `final/errors.json`：所有动作与参数 FP/FN。
- `final/freeze.json`：测试前冻结的模型和配置。
- `final/audit.json`：独立复核，含从原始标注重建全部转换记录。
- `act/best.pt`、`arg/best.pt`：选定模型；各目录保存逐轮成绩和训练配置。
- `data/procedural/ricoreca/converted/split.json`：精确文档划分。

核心入口：`prepare_ricoreca_data.py` → `prepare_ceasdrl_holdout.py` →
`train_ceasdrl_holdout.py` → `evaluate_ceasdrl_holdout.py` → `audit_ceasdrl_ricoreca.py`。
训练命令的实际参数保存在两个 `config.json`，必须复用相同 converted 数据和特征 manifest。
新文本预测使用 `scripts/predict_ceasdrl_holdout_text.py`，指定本次 run 目录、UTF-8 文本和离线 py36 词向量环境。

```powershell
python scripts/predict_ceasdrl_holdout_text.py --run results/ceasdrl_ricoreca_64_16_20_seed0 --text-file your_text.txt --output results/ricoreca_prediction --legacy-python C:/Users/Apexmod/miniforge3/envs/py36/python.exe
```

仅含原文的 `plain_text_smoke_input/1recipe142.txt` 已独立运行，9 个动作及其参数输出与批量测试完全一致。
核对记录见 `plain_text_smoke/consistency_check.json`；该验证没有把动作或参数标签传入预测入口。

官方数据/代码：[RiCoRecA](https://github.com/FilipposVentirozos/RiCoRecA)。
本地压缩包及逐文件校验记录在 `data/procedural/ricoreca/archive_receipt.json`。
