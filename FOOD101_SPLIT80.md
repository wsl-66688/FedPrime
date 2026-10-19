# Food101 固定分层 80:20 版本

日期：2026-10-18。

## 已完成的结果

- 保留发布代码实际采用的 8 类、6,727 条有效图文样本，原始验证集的 548 条不加入。
- 新划分为 **5,381 条训练、1,346 条测试**。测试数取 `ceil(6727×0.2)`，因此比例因整数取整略有差异。
- 固定划分种子 0，按类别进行分层；各类测试数使用最大余数分配，和其 20% 目标相差不足 1 条。
- 使用版本化的 SHA256 排序选样，避免依赖 Python/NumPy 的随机状态；完整样本 ID 保存在清单中。
- 三个缺失场景共用同一份 20 客户端划分，每个客户端 269～270 条，全部训练索引覆盖一次、无重叠。
- 缺失模式也固定为种子 0，使用独立的 JSON 记录，绑定新样本 ID 和顺序，所有方法复用。
- 逐条对照确认 6,727 条样本的图片字节、文本、标签保持一致，仅改变训练/测试归属及排列；新旧文件分开保存。
- 13 项相关测试通过，包括实际数据读取类、任务读取入口、缺失记录加载、划分覆盖和结果存储/汇总。

**这是自建的分层 80:20 baseline，仍保留发布实现的 8 类。它没有确认论文的“最频繁 8 类”、6,728 条样本及作者原始划分，不能称为严格还原 Table 1。**

模型结构、分类器输出维度、损失、优化器和聚合算法沿用现有代码。本次没有添加 checkpoint/断点续训，也没有在本地启动模型训练。

## 版本和文件位置

协议标识：`released8_stratified80_seed0_v1`。

小型、可版本管理的完整划分清单：

```text
data_protocols/released8_stratified80_seed0_v1.json
```

新数据及缺失记录：

```text
benchmark/RAW_DATA/FOOD101/protocols/released8_stratified80_seed0_v1/
├── split.json
├── prepared-files.json
├── preparation-report.json
├── generate_arrows/
│   ├── protocol.json
│   ├── food101_train.arrow
│   └── food101_test.arrow
├── missing_tables/        # 三种训练/同分布测试条件；固定 *.pt.json
└── other_tests/           # 完整模态、仅文本、仅图像等固定测试条件
```

本地新 Arrow 大小分别为 455,181,130 字节、116,245,898 字节，合计约 571 MB。远程不同 PyArrow 版本可能导致字节布局略有不同，以对应生成记录中的校验值和完整样本 ID 核对。

新任务位于 `fedtask/`，名称均带后缀 `_protocol_released8_stratified80_seed0_v1`，与旧任务区分。原始 Arrow、旧 `.pt`、旧任务及旧实验结果保留。

## 需要上传哪些小文件

已打包为项目根目录的 `FOOD101_SPLIT80_UPLOAD.zip`，只含下列代码、清单和说明，不含数据集与权重：

- `utils/food101_protocol.py`：划分、任务及固定缺失记录的验证逻辑。
- `notebook/food101_split80.py`：新协议的 prepare/check/run/summary 入口。
- `notebook/food101_table1.py`：为运行和归档逻辑增加协议上下文；默认旧入口仍采用旧划分。
- `benchmark/food101/base_dataset.py`：识别新协议，完整读取 5,381/1,346 条，旧协议保留原前缀规则。
- `benchmark/food101/core.py`：按任务指定路径读取新数据及全部测试条件。
- `benchmark/food101/dataset.py`：按新样本顺序使用固定 JSON 缺失记录，旧任务保留 `.pt` 路径。
- `utils/logger/table1_logger.py`：独立结果目录。
- `data_protocols/released8_stratified80_seed0_v1.json`：固定样本清单。
- 本说明。

本地不需要安装完整训练环境。若旧的长训练正在运行，保留其运行及结果，待安全的任务边界再部署这批代码；不要在运行中切换数据协议。

## 远程第一步：上传并解压小包

上传 `FOOD101_SPLIT80_UPLOAD.zip` 到：

```text
/home/dministrator/Downloads/FedPrime-main/FOOD101_SPLIT80_UPLOAD.zip
```

在远程 SSH 终端执行：

```bash
cd /home/dministrator/Downloads/FedPrime-main
conda activate fedprime
python -m zipfile -e FOOD101_SPLIT80_UPLOAD.zip .
```

结果：代码和清单解压到项目内相同相对位置；不覆盖旧数据和结果目录。此步骤不生成训练结果，**不需要下载到本地**。

## 远程第二步：生成新数据、三个任务及固定缺失记录

```bash
python notebook/food101_split80.py prepare
```

使用远程已有的三个 JSON 元数据和两份完整旧 Arrow，生成上述 `protocols/released8_stratified80_seed0_v1/` 目录，以及三个带协议后缀的新任务。无需重新下载图片、权重，也无需上传本地新 Arrow。额外数据空间约 0.6 GB，准备时另需少量临时空间。

成功标准：输出 `train` 5,381、`test` 1,346，三个任务 `exists: true`，`errors: []`。重复 prepare 会核对已完成版本，遇到不一致会拒绝覆盖；若生成中断，可能保留 `.prepare_*` 临时目录用于排查，不会替换旧数据。

**下载需求：**无需下载新 Arrow；如需我核对准备情况，只下载新协议目录中的 `preparation-report.json` 即可。

## 远程第三步：检查项目环境和新数据

```bash
python notebook/food101_split80.py check --runtime --gpu 0 > food101_split80_remote_check.json
```

结果保存在：

```text
/home/dministrator/Downloads/FedPrime-main/food101_split80_remote_check.json
```

成功标准：`errors: []`；三个任务存在；有效行数 5,381/1,346。保留“不是作者确认的划分”这一提示是正常的。

**请将这个 JSON 下载到本地项目根目录**，方便核对；大数据文件不需要下载。

## 远程第四步：重新完成新协议的一轮验证

旧协议的 smoke 不能替代新协议验证。先运行：

```bash
python notebook/food101_split80.py run --stage smoke --gpu 0 --seeds 1234
```

结果位于远程：

```text
/home/dministrator/Downloads/FedPrime-main/experiments/table1_split80/released8_stratified80_seed0_v1/日期_smoke_唯一编号/
```

成功标准：对应子目录 `status.json` 为 `complete`、`rounds` 为 1，且存在 `metrics.json`。该次只运行 FedPrime/Miss Both；正式结果汇总会排除 smoke。

**请下载该次完整 smoke 目录到本地相同相对位置**，便于检查日志和参数。无需下载新 Arrow、权重和原始图片。

## 后续正式实验：使用新入口

新协议 smoke 通过后，在远程的 tmux 或服务器作业会话内执行：

```bash
python notebook/food101_split80.py run --stage fedprime --gpu 0 --seeds 1234
```

仍是三个缺失场景、各 250 轮，共 750 轮；训练参数沿用既定方案。结果保存在上述 `experiments/table1_split80/released8_stratified80_seed0_v1/` 下对应 `日期_fedprime_唯一编号/`。

**运行期间不必下载；结束后下载该次完整结果目录到本地相同相对位置。** 若失败，也下载失败目录的日志和状态用于排查。当前仍无完整断点恢复功能。

完整五种方法、三个 seed 的实验可将 `--stage` 改为 `all`、`--seeds` 改为 `1234 1235 1236`；共 45 次训练，先完成小规模验证后再安排。

将下面占位目录换成实际的一个新协议批次目录，可在远程汇总：

```bash
python notebook/food101_split80.py summary --results experiments/table1_split80/released8_stratified80_seed0_v1/实际批次目录
```

在该批次目录生成 `table1-runs.csv`、`table1-summary.csv`；**下载这两份 CSV 到本地分析**。仍采用最后一轮准确率，报告均值和样本标准差；不混入旧协议结果。

**注意：原来的 `python notebook/food101_table1.py ...` 命令仍运行旧协议。要用新的 80:20 数据，必须使用 `food101_split80.py`。**
