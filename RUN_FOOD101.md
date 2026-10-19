# Food101 本地与远程运行步骤

本说明对应当前项目目录结构。

## 当前工作方式：本地编辑，远程运行

本地用于让 Codex 修改代码，再将修改上传到远程项目；训练在远程的 `fedprime` 环境运行。采用此方式时，本地无需安装完整训练环境，第 3～6 节仅供需要本地执行时使用。你已确认远程准备完成，不需要因本地修复而重新下载或重建远程数据。

2026-10-17 在本地发现两个 Arrow 文件被截断到约 16 MB、缺少文件尾部。本次修复将预处理改为每批 256 条写出，完成格式和行数检查后再替换正式文件；样本顺序、合并训练/验证集、标签映射和跳过无文本样本的规则不变。两个正则表达式也改为原始字符串，以消除 `SyntaxWarning`。

原始截断文件保留在 `benchmark/RAW_DATA/FOOD101/generate_arrows/_corrupt_backups/before_rebuild_jh3ndsi9/`。仅凭截断文件无法确定是哪次传输或写入导致，不应据此断言具体原因。

建议在 PyCharm 的 `Deployment → Configuration → Excluded Paths` 中排除以下本地目录，让代码同步与运行资源分开：

- `benchmark/RAW_DATA/FOOD101`
- `benchmark/pretrained_model_weight`
- `fedtask`

本次没有自动修改 PyCharm 配置，也没有更改远程文件。现有 `all_models.sh` 的路径拼写、算法、GPU 和 W&B 设置仍保持原样；运行时使用远程已经确认的配置，或使用第 6 节的直接 Python 命令并调整 GPU。

## 1. 预训练资源的保存位置

在项目根目录下应有：

```text
benchmark/pretrained_model_weight/
├── vilt_200k_mlm_itm.ckpt
└── bert-base-uncased/
    ├── config.json
    └── vocab.txt
```

ViLT 权重来自项目 `env_setup.sh` 指定的官方 GitHub Release。BERT 词表和配置来自 Hugging Face 的官方历史存储。当前代码使用 `BertTokenizer.from_pretrained(...)` 加载此目录；这里准备的是分词器所需文件，无需另外下载完整的 BERT 模型权重。

原始图片、JSON 标注、Arrow 数据、预训练资源是不同内容。在哪台机器运行，就需要在那台机器准备对应资源。

## 2. 在远程 Linux 补齐资源

在目标服务器的 SSH 终端执行。确认下载文件没有被 PyCharm 的上传任务同时写入。

```bash
cd ~/Downloads/FedPrime-main
mkdir -p benchmark/pretrained_model_weight/bert-base-uncased

curl -fL --retry 3 --connect-timeout 20 -C - \
  "https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt" \
  -o benchmark/pretrained_model_weight/vilt_200k_mlm_itm.ckpt

curl -fL --retry 3 --connect-timeout 20 \
  "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt" \
  -o benchmark/pretrained_model_weight/bert-base-uncased/vocab.txt

curl -fL --retry 3 --connect-timeout 20 \
  "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json" \
  -o benchmark/pretrained_model_weight/bert-base-uncased/config.json
```

每条下载命令都必须成功退出。下载失败时先处理报错，再进行后续步骤。

在已安装项目依赖的 `fedprime` 环境中验证分词器：

```bash
conda activate fedprime
python -c "from transformers import BertTokenizer; t=BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased', local_files_only=True); print('vocab_size:', t.vocab_size); print(t('a photo of food')['input_ids'])"
```

预期 `vocab_size` 为 `30522`。ViLT 权重约为 1.63 GB，文件完整后才能训练。

## 3. 本地 Windows 配置环境（仅本地执行时需要）

此前检查到 Windows 的默认 Python 是 3.13，本地 Conda 环境列表没有 `fedprime`。远程已经存在的环境不会自动成为本地环境。当前只在远程训练的工作方式无需完成本节。

以下命令在 Windows PowerShell / Anaconda Prompt 执行。按 README 使用 Python 3.8，已有同名环境时跳过创建。

```powershell
cd E:\桌面\FedPrime\FedPrime-main
conda create -n fedprime python=3.8 -y
conda activate fedprime
python --version

python -m pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
python -m pip install -r requirements.txt
python -m pip install protobuf fonttools imgaug opencv-python pyyaml regex scipy
```

依赖版本沿用仓库的 `env_setup.sh` 和 `requirements.txt`。如安装失败，先处理实际报错；这些步骤尚未在新建的本地环境中验证。

验证当前解释器、GPU 和 BERT 分词器：

```powershell
python -c "import sys,torch; from transformers import BertTokenizer; print(sys.executable); print('CUDA available:',torch.cuda.is_available()); t=BertTokenizer.from_pretrained('benchmark/pretrained_model_weight/bert-base-uncased', local_files_only=True); print('vocab_size:',t.vocab_size)"
```

如果使用 PyCharm 本地运行，项目解释器选择本地 `fedprime`，工作目录选择 `E:\桌面\FedPrime\FedPrime-main`。如果选择远程解释器，执行的仍是远程环境。

## 4. 在本地转换 Arrow 数据

本地图片和三个 JSON 已准备好。本次已在本地重新执行转换；以后需要重新生成时，使用下面的相同命令。脚本分批写入临时文件，通过检查后才替换对应的正式文件。

```powershell
python notebook/make_arrow_food101.py
Get-ChildItem .\benchmark\RAW_DATA\FOOD101\generate_arrows\
```

应生成 `food101_train.arrow` 和 `food101_test.arrow`。根据当前标注，训练集为 67,972 条，测试集为 22,716 条。原脚本会跳过 16 个没有文本的训练样本。

转换会读取大量图片并占用磁盘，默认每批处理 256 条以控制内存。已有完整 Arrow 时可以跳过转换。新文件包含多个 RecordBatch，文件字节数可能与旧版单批输出略有不同，读取方式和数据列保持一致。

## 5. 在本地生成 20 客户端任务

以下直接调用 Python，适用于 PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path fedtask,benchmark/RAW_DATA/FOOD101/missing_tables_8_classes,benchmark/RAW_DATA/FOOD101/missing_tables_other_tests | Out-Null
python generate_fedtask.py --benchmark food101 --dist 0 --skew 0 --num_clients 20 --seed 0 --missing --missing_ratio_train 0.7 --missing_ratio_test 0.7 --missing_type_train both --missing_type_test both --both_ratio 0.5 --max_text_len 40
```

只有该命令成功后，才进入训练。

## 6. 本地运行 Fed-Prime

先使用 1 轮运行检查流程。下面默认使用 CPU，未启用 W&B；CPU 上训练与完整评估都可能很慢。

```powershell
python main.py --task food101_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5 --model Fed-Prime --algorithm multimodal.food101.Fed-Prime --sample full --aggregate other --num_rounds 1 --proportion 1.0 --lr_scheduler 0 --seed 1234 --learning_rate 0.05 --num_epochs 1 --pool_size 20 --top_k 5 --num_outer_loops 5 --learning_rate_decay 1.0 --batch_size 4 --test_batch_size 4 --max_text_len 40
```

如果上面的 CUDA 检查为 `True`，且准备使用第一张可见显卡，在训练命令末尾添加 `--gpu 0`。显存不足时调整 batch size。正式复现实验时再按实验设置恢复轮数、批量大小和其他参数；这里的 1 轮小批量命令不是论文完整配置。

远程 Linux 在生成相同任务后，也可以使用相同的 `python main.py ...` 命令，GPU 编号按服务器实际可见设备选择。

仓库现有 `all_models.sh` 默认算法为 `Fed-Intra`、GPU 为 `1`，并且内部引用 `script/`；实际目录为 `scripts/`。本说明直接指定 `Fed-Prime` 和命令参数，不依赖该脚本。

## 7. 已完成工作与验证边界

- 数据和预训练资源已补齐，预训练资源来源及校验值记录在 `benchmark/pretrained_model_weight/RESOURCE_INFO.json`。
- 本次修改 `notebook/make_arrow_food101.py`，增加分批写入、临时文件替换和输出验证；模型、训练算法和实验参数未修改。
- `notebook/test_make_arrow_food101.py` 验证跨批次内容、缺图失败时保留原文件、全部缺文本时生成有效空文件三种情况。
- 已执行本地 Arrow 重建；现有联邦任务和缺失记录做一致性核验，不重新生成随机划分。
- 没有为本地安装完整训练环境，也没有在本地启动训练。本地只编辑、远程运行时，这些不属于缺失的准备工作。
- 本地数据检查不能替代远程的实际训练验证；远程已准备完成的文件无需重跑。
