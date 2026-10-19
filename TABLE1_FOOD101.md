# Food101 Table 1：当前缺口与运行步骤

> 2026-10-18 已另行实现自建分层 80:20 版本（5,381/1,346），操作见 `FOOD101_SPLIT80.md`。本文原命令仍保留旧协议（4,904/1,823），不要混用结果。

依据项目根目录上一层的 `FED-PRIME.pdf`：正文 4.1、Table 1、补充材料 B（PDF 第 6、7、12、13 页）。

## 连续运行 1～2 天前的检查结论

2026-10-17 再次检查：本地 Arrow 结构、有效类别、预训练资源大小和已有 Both 任务覆盖通过，记录在 `FOOD101_PRELAUNCH_AUDIT.json`。本地没有完整训练环境，无须为这次检查安装。

随后已收到用户提供的远程终端输出：解释器为 `/home/dministrator/.conda/envs/fedprime/bin/python`，Python 3.8.20，PyTorch 2.4.1+cu121，CUDA 构建版本 12.1，CUDA 可用。GPU 报告空闲显存 28,712 MiB（约 28.0 GiB），系统可用内存 24 GiB，项目与 `/tmp` 所在的同一磁盘剩余 763G。远程解释器及资源基础检查已通过；显存空闲量是当时快照，实际模型峰值仍需短跑验证。

当前 PyTorch/CUDA 与 `env_setup.sh` 指定的 2.3.1+cu118 不同，应保留该版本差异记录。

**远程一轮验证现已确认成功。** 本地已同步 `experiments/table1/20261017_210820_smoke_a1776518/`，其中 `both_Fed-Prime_seed1234/status.json` 为 `complete`，轮数为 1，`metrics.json` 包含五种目标测试准确率，日志包含服务器聚合、最终评价及结束标记。实际使用 CUDA 0、20 客户端、batch 512、学习率 0.05、提示池 20、每池 top-k 5。该次训练及评价计时 362.6102 秒，约 6 分钟；该结果证明 Miss Both 的完整一轮流程跑通，尚未证明三个场景的完整 250 轮稳定性或论文数值已复现。

`run --stage fedprime --gpu 0 --seeds 1234` 会依次运行 **Both、Text、Image 三个场景，每个 250 轮，共 750 轮**。建议先完成下面几项：

1. 在远程确认新入口和 `utils/logger/table1_logger.py` 都已同步，工作目录与解释器正确。运行 `check --runtime --gpu 0`，检查 `errors` 为空。
2. 在远程提前生成 Text、Image 两个任务，再检查 `tasks` 中三个 `exists` 都为 `true`。入口原本会在训练开始前自动生成缺少的任务，提前生成是为了先拿到完整准备记录。
3. 已确认远程一次 `smoke` 成功，且本地可查看 `status.json`、`metrics.json` 和完整日志。1 轮验证能检查完整流程，但不能保证之后所有轮次都稳定。
4. 查看实际 GPU 空闲显存、系统内存和磁盘空间；本地资源检查不能代替远程检查。不要在长跑前无记录地更新依赖。
5. 训练期间固定数据、任务和源文件。当前本地 PyCharm 配置为自动上传 Always，只排除了损坏文件备份目录；建议暂停自动上传，或为原始数据、权重、`fedtask`、`experiments` 配置排除，后续按需上传代码。
6. 根据前文，明确本次是公开实现 baseline；作者精确数据协议及结果选择规则仍未确认。

可在远程执行以下准备命令。每条都检查是否成功，失败时先处理错误，不继续正式训练：

```bash
cd ~/Downloads/FedPrime-main
conda activate fedprime
python notebook/food101_table1.py check --runtime --gpu 0

bash scripts/food101/main_exp_20clients/miss_text/20cl.sh
bash scripts/food101/main_exp_20clients/miss_img/20cl.sh
python notebook/food101_table1.py check --runtime --gpu 0

nvidia-smi
free -h
df -h .
python notebook/food101_table1.py run --stage smoke --gpu 0
```

### 中断保护目前的实际限制

- 日志持续保存，但当前算法是在每个场景的训练结束后调用正式指标保存；运行到中途时没有 `metrics.json` 不一定代表异常。
- 现有实现没有保存可恢复完整训练状态的周期性 checkpoint，也没有 resume 参数。服务器重启或进程崩溃后，不能保证从中断轮次继续。
- 已完成场景的结果保存在其独立目录中，但重新执行整条命令会新建批次并从头运行，不会自动跳过已完成场景。

长训练可以在服务器作业系统或 tmux 中启动。tmux 支持断开和重新连接终端，能避免 SSH 连接断开直接影响其中的任务；它不能代替模型 checkpoint，也不能应对服务器重启。[tmux 官方说明](https://github.com/tmux/tmux/wiki/Getting-Started)

远程已安装 tmux 时，先创建会话：

```bash
tmux new -s fedprime_food101
```

然后在该会话中重新确认目录和 Conda 环境，再启动已通过检查的正式命令。按 `Ctrl+b` 后按 `d` 可以脱离会话；之后重新登录并用 `tmux attach -t fedprime_food101` 返回。若训练已经在普通终端运行，不要为了切换会话直接结束现有任务。

若要求“进程中断后接着训练”，还需要单独实现并验证完整断点恢复，包括动态提示池、客户端状态及随机数状态；只保存提示向量或一份模型权重不能证明等价续训。本次检查没有修改正在使用的训练入口和算法。

## 目前缺什么

1. **
2. **本地只存在 Miss B论文数据协议尚未对齐。** 论文写 8 个最频繁类别、6,728 条、80:20；当前代码截取 Arrow 前部，得到 4,904/1,823 条。不能直接改成任意 80:20 后声称还原作者划分。需要作者类别/样本 ID、划分文件或进一步说明。当前入口明确采用“公开数据子集＋论文训练参数”，不自动改变数据和算法。oth 任务。** Miss Text、Miss Image 的任务可由新入口在远程生成；已有任务保留，并核对覆盖范围和缺失参数。远程可能已经有这些任务，应以远程检查为准。
3. **原批量脚本不能直接作为正式配置。** 新入口绕过路径拼写错误，固定 batch 512，并为 Fed-Intra/Fed-Inter 指定 top-k 10；FedPrime 每池 top-k 5。
4. **需要远程实际训练证据和 GPU 检查。** 本地不安装训练环境；远程使用 `fedprime`。已有数据和权重无需再次上传。脚本检查 CUDA 和基础依赖，完整兼容性仍由 1 轮运行验证。
5. **需要独立结果和汇总。** 新 logger 每次写入唯一运行目录；汇总最终轮的五种测试准确率，不使用逐列最高测试分数。作者具体模型选择规则、重复次数和 ± 的定义仍未确认。

## 本次新增文件

- `notebook/food101_table1.py`：检查、生成缺失任务、运行实验、保存环境与输入标识、输出 CSV 汇总。
- `utils/logger/table1_logger.py`：只改变结果保存位置，沿用已有评价流程。
- `notebook/test_food101_table1.py`：本地测试指标提取、汇总和独立存储。
- 本说明。

没有修改数据子集、客户端算法、模型结构或旧批量脚本。检查只读取 Arrow 结构和标签，不重新遍历所有图片内容；资源首先按已知大小检查，不能把这理解为完成了全部文件的哈希校验。

## 第一步：上传小文件

将上述文件上传到远程 `~/Downloads/FedPrime-main` 下相同相对位置。特别注意 logger 文件也要上传。不要让本地 `fedtask` 覆盖远程已有任务。

## 第二步：远程检查

在远程 SSH 终端执行：

```bash
cd ~/Downloads/FedPrime-main
conda activate fedprime
python notebook/food101_table1.py check --runtime --gpu 0
```

`errors` 应为空。`warnings` 中的数据协议差异会保留；缺少 text/image 任务时会提示，运行对应实验时可生成。GPU 0 是当前进程可见的第一张卡，可按服务器实际情况调整。

当前本地已检查：Arrow 有 67,972/22,716 条，代码实际选取 4,904/1,823 条；权重和词表大小符合现有资源记录；已有 Both 任务覆盖正确。远程需要执行上述命令另行验证。

## 第三步：运行 1 轮 FedPrime 验证

```bash
python notebook/food101_table1.py run --stage smoke --gpu 0
```

只运行 Miss Both、FedPrime、seed 1234、1 轮；batch 保持 512，检查加载、本地更新、聚合及全部测试能否完成。`smoke` 会被正式汇总排除。

输出到 `experiments/table1/日期_smoke_唯一编号/`。每次执行使用独立目录；终端会打印确切路径。

显存不足时保留错误，不直接把小 batch 实验与论文配置混称。本地每客户端约 249 条，batch 512 实际最多取该客户端数据；显存仍需通过远程运行确认。若需调整训练 batch 或其他优化策略，应建立单独适配实验。

## 第四步：完成 FedPrime 的三种训练条件

```bash
python notebook/food101_table1.py run --stage fedprime --gpu 0 --seeds 1234
```

共 3 次训练，每次 250 轮，分别为 Miss Both、Miss Text、Miss Image，训练和同分布测试缺失率均为 0.7。

- 20 客户端，每轮全参与；本地 1 epoch。
- SGD，学习率 0.05，训练/测试 batch 均为 512。
- 提示池初始大小 20，每池选 5；最大文本长度 40。
- `num_outer_loops=5`、恒定学习率设置沿用发布脚本；这些实现取值不能全部称为论文明确给出的参数。
- 优化器动量、权重衰减等未显式覆盖的设置沿用当前 `main.py` 默认值，实际生效参数保存在结果 `meta` 中。

先看这一组运行是否合理，再扩展其他方法。若需要先看完整命令、不训练，加 `--dry-run`。

## 第五步：完整主表比较

```bash
python notebook/food101_table1.py run --stage all --gpu 0 --seeds 1234 1235 1236
```

这是 **45 次完整训练**：5 种方法 × 3 种训练缺失条件 × 3 个 seed，按顺序执行。3 个 seed 是本方案建议，不是已确认的作者设置；可以先只用一个 seed 检查整套流程。不同方法复用对应场景的相同任务及缓存。

五种方法：FedPrime、FedAvg-P、FedMSplit-P、Fed-Intra、Fed-Inter。

- FedPrime 两池各选 5；Fed-Intra/Fed-Inter 单池选 10。
- FedAvg-P/FedMSplit-P 沿用其模型中前 6 层、提示长度 16 的实现。
- 每次训练沿用现有代码的评价入口，记录同训练分布、Miss Both、完整模态、仅文本、仅图像；部分场景还会额外记录其他测试项。
- Miss Both 训练下，前两列相同，汇总保留相同值；整理成论文样式时可将重复列写为 `-`。

脚本不会自动跳过以前成功的相同实验，也不支持训练断点恢复。`all` 是独立的新运行，若先前做过 FedPrime，会重新运行它。45 次实验不要在 1 轮验证通过前启动；使用服务器作业系统或可靠持久会话避免 SSH 断开影响任务。

## 第六步：汇总一个完整实验批次

将下面占位路径换成该次运行终端打印的实际目录：

```bash
python notebook/food101_table1.py summary --results experiments/table1/实际运行目录
```

生成：

- `table1-runs.csv`：每次运行最终轮的准确率，单位为百分数。
- `table1-summary.csv`：按训练场景、方法、测试条件计算均值和样本标准差，列出 seed 和运行次数。

只汇总成功结束的 250 轮记录，排除 1 轮验证及失败任务。只有一个 seed 时标准差留空。不会从多个轮次中挑选最高值。当前结果选择规则为本方案制定的“最后一轮”，仍需和作者原始规则确认；达到类似数字不自动证明严格复现。

每个运行目录保存命令、状态、控制台输出和 `metrics.json`；批次目录保存环境、代码哈希及 `source.zip` 源文件快照、任务与缺失缓存哈希。这里固定客户端划分和缺失缓存，只改变训练 seed；不同随机划分的稳定性属于后续实验。

## 如何解决数据协议仍不一致的问题

### 2026-10-17 进一步核查：差异已定位到类别和划分规则

已将 Arrow 中实际读取的全部样本 ID 与原始三个 JSON 元数据逐条对照，确认：

- 当前训练的 4,904 条，正好等于 `split.json` 中类别编号 0～7 的全部原始 `train` 样本，顺序也一致。
- 当前测试的 1,823 条，正好等于同样 8 类的全部原始 `test` 样本，顺序也一致。
- 这 8 类还有 548 条原始 `val` 样本。转换时它们被追加到所有类别的训练数据之后，因此前缀切片没有使用它们。
- 这 8 类的样本都有文本。此前跳过的 16 条无文本记录不能解释这里与 6,728 相差 1 条。
- 使用这 8 类的 train/test 共 6,727 条；连同 val 共 7,275 条。直接合并 val 也不会得到论文的 6,728 条或 80:20。
- 以当前元数据中全部有效图文对的数量统计，频率最高的 8 类合计 7,542 条，与当前类别编号 0～7 的集合不同。

因此不能通过补一条数据、合并验证集或简单改分割比例，确认还原论文协议。这次核查证明当前切片恰好选中固定的完整类别子集；它仍无法证明作者生成 Table 1 时实际采用了哪份数据。

本地保存了两份证据：

- `FOOD101_DATA_PROTOCOL_AUDIT.json`：原始 JSON 校验值、各类数量、Arrow 对照结果。
- `FOOD101_RELEASED_SAMPLE_IDS.json`：当前使用的全部训练/测试样本 ID，以及被排除的验证样本 ID。用于冻结当前 baseline，不会自动替换数据加载规则。

### 推荐处理顺序

可以先完成上述公开实现实验，建立可追溯的 baseline。同时查找作者固定的类别列表、样本 ID 和划分文件；必要时提出具体问题：为什么发布代码用前缀切片得到 6,727 条和约 72.9:27.1，而论文说明 6,728 条和 80:20？

拿到确认后另建数据协议，统一更新样本映射、类别映射、任务索引及缺失记录，并让所有方法重跑。保留原实验，不能将两种协议的结果混入同一比较组。没有作者划分时，可以透明报告按论文文字构建的适配协议，但不能保证还原原文精确数值。

如果暂时没有作者答复，可以采用以下两个明确区分的实验：

1. **公开实现 baseline**：保留 4,904/1,823 及现有索引、缺失缓存，运行论文训练参数。适合先验证系统、学习方法，并与自己的改进做一致条件下的比较。
2. **按论文文字构建的适配实验**：对有效图文样本统计频率、固定 8 类，使用记录明确的种子做分层 80:20 划分，保存样本 ID；随后新建数据、任务和缺失记录。当前样本数应由统计结果决定，不能为凑 6,728 而任意删数据。该实验单独报告，不能直接称为还原 Table 1 的原始划分。

建议先只验证一个公开实现设置；数据协议确认前不必立即投入全部 45 次正式训练。

本地只验证了输入检查、命令生成和记录/汇总逻辑；尚未在本地或远程执行完整模型训练。资源是否满足、模型运行是否成功，仍以你的远程 1 轮检查及日志为准。

### 可提交给作者的问题草稿

下面是本地草稿，供你在作者仓库的 Issues 页面核对并提交。问题讨论入口：https://github.com/hangpt01/FedPrime/issues

**Title: Clarification of the Food101 subset and split used in Table 1**

Thank you for releasing FedPrime. I am reproducing the Food101 results in Table 1 and would appreciate clarification of the data protocol.

Section 4.1 describes the eight most frequent classes and an 80:20 train/test split; Supplement B reports 6,728 samples. With the Food101 JSON metadata linked in DATA.md, I obtain 67,972 train and 22,716 test Arrow rows after the released conversion procedure. The prefix selection in base_dataset.py then uses 4,904 train and 1,823 test rows, or 6,727 in total.

Comparing every selected image ID with split.json, these are exactly the original train/test samples of class IDs 0 through 7. Another 548 validation samples of these classes are outside the selected prefixes. None of these selected-class samples lacks text. Counting all valid image-text pairs also gives a different set of eight most frequent classes.

Could you please clarify:

1. The exact class names and sample IDs used for Table 1, and the source of the 6,728 count.
2. Whether the experiments used the original train/test partitions or a new 80:20 split, and whether validation samples were included.
3. Whether the current prefix-selection logic matches the data used for the reported results.
4. Whether you could share the sample split, client partitions, or missing-modality masks used in those experiments.

I can provide the per-class counts and selected sample-ID manifest if helpful. Thank you.
