# RiCoRecA cEASDRL 实验

更新：2026-11-22。训练、封闭测试和独立复核均已完成。测试动作名称 F1 **96.85%**，对应参数名称 F1 **61.80%**。

完整报告：`../../../results/ceasdrl_ricoreca_64_16_20_seed0/REPORT.md`。
两个阶段各训练 10 轮，按验证集分别选中第 10、9 轮；动作阶段使用 CPU，参数阶段使用 CUDA。

## 数据与划分

用户提供 `data/data.zip`，SHA256：`921f22602f1af88d21bb40038204f9e0bf52214c866ac5a889a277901f12ba2c`。
逐文件校验见 `archive_receipt.json`。主标注采用 Annotator_P。

原始 JSON 有 163 篇；作者已发布的 Cross_Val CSV 合计覆盖 156 篇。
本实验使用这 156 篇的并集作为语料范围，重新划分，不使用作者的旧折分。
7 篇不在发布语料中的原始记录，其 ID 保存在 `converted/split.json`；其中 2 篇原始 answer 为 ignore。
全部 raw 原始 ACTION 共 2,290 个；发布语料对应的 raw ACTION 为 2,177 个，发布后的标注保留 2,176 个。
论文统计与实际训练目标不能直接混用。

| 集合 | 文档 | 动作 | 动作—参数关联 |
|---|---:|---:|---:|
| train | 99 | 1,313 | 2,616 |
| dev | 25 | 378 | 738 |
| test | 32 | 485 | 936 |
| 合计 | 156 | 2,176 | 4,290 |

按文档 ID 排序后，`train_test_split(test_size=.2, random_state=1)` 留出 test；
对剩余部分再次使用相同参数留出 dev。名义比例 64/16/20，整数比例约 63.46/16.03/20.51。
同一篇所有动作、参数均留在同一集合。不是五折或十折，也不是句子随机拆分。

## 目标和转换

只提取完整 ACTION 名称和与该动作关联的完整显式参数名称；不预测 ES/EX/OP 或语义角色。
参数保留食材、工具、指代词、部分实体，以及与 ACTION 关联的状态、计量、设置和目的等文字片段。
指代词按原文提取，不替换为人工补充或推断的对象名称。
IF/UNTIL/REPEAT 不作为 ACTION；仅属于这些控制谓词的 662 个参数 mention 不计入本任务。
因此这不是 RiCoRecA 全部实体、控制流、共指和隐式实体任务的复现。

作者 CSV 用动作文字表示 membership，同篇重复动词可能歧义。
转换同时使用发布的 membership 和原始 Prodigy 位置关系：沿 Member/Or/Modifier 找到谓词即停止，
根据 Join 共享参数；Dependency 不参与参数归属。每条输出关联记录原始关系路径。
任何缺失或歧义都会使转换报错，不通过最近动词猜测。
保留的 4,290 条关联全部能够唯一定位；全文和名称位置精确匹配，BIO 往返无损。
最多 400 tokens，全部目标都在 600-token 动作输入和 100-token 参数窗口内，没有截断目标。
发现 1 条指向未标注词的失效原始关系，已逐条记录，不用于生成关联；所有保留目标仍能解析。

## 方法和评估

沿用已核对的 PyTorch cEASDRL CNN/DQN 与冻结 GloVe-100 + ELMo-small-768，两个阶段从头训练。
三分类 BIO 用于完整边界，balanced replay 与 mention-rate shaping 沿用 RISeC 改编版。
这是 cEASDRL 方法的任务适配，不应声称复现了原论文的全部设定或分数。
训练 seed 0，最多 10 epochs，patience 3，Adam lr .001；分别按 dev exact-span micro F1 选 checkpoint。

训练 feature manifest 只含 train/dev，test 为空。两个 checkpoint 完成后写入 freeze.json，
再用测试文本预测动作，并以预测动作建立参数输入。测试动作、参数标签不参与解码。

- 动作 F1：完整动作名称及其出现位置都匹配。
- 对应参数 F1：动作实例及参数名称/位置同时匹配；漏掉动作时，其参数仍计入 FN。
- 全部文档汇总 TP、预测数量和 gold 数量，计算 micro precision/recall/F1。
- 使用独立的字符位置评分器复算，必须与 token/name 评分器完全一致。

只有一个固定划分和训练种子，不给交叉验证均值或标准差。
RiCoRecA 与 RISeC 的标注范围、语料和划分不同，跨数据集 F1 不足以证明数据集质量高低。
后续 LLM/NL2P 对比应使用同一份 JSONL、split 和评分器。

## 文件与复现入口

- `converted/{train,dev,test}.jsonl`：转换后的完整目标，含关联来源。
- `converted/split.json`：全部文档 ID 和源文件哈希。
- `converted/conversion_audit.json`：计数、覆盖检查和代码哈希。
- `features_train_dev/manifest.json`：训练及验证特征，不含测试样本。
- `experiment_64_16_20.json`：协议和当前状态。
- `../../../results/ceasdrl_ricoreca_64_16_20_seed0/`：本次训练与最终评估目录。

数据准备：`python scripts/prepare_ricoreca_data.py`。
特征入口：`scripts/prepare_ceasdrl_holdout.py`（新词向量需离线 py36 环境）。
训练入口：`scripts/train_ceasdrl_holdout.py`。
统一封闭测试入口：`scripts/evaluate_ceasdrl_holdout.py`。
现有 RISeC 模型、训练脚本和结果未覆盖。

## 来源

- 官方仓库：https://github.com/FilipposVentirozos/RiCoRecA
- 核对 commit：af80d0c3fdaa5a375625bb9f006ce536770068f2
- 论文：https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1550604/full
- 数据：https://figshare.com/articles/dataset/22317796
- 标注指南：https://figshare.com/articles/online_resource/25853902

来源核查及此前网络获取失败记录保存在 `source_review.json`；本次实际使用用户提供的本地压缩包。
