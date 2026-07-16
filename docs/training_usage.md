# 重构后训练与数据流水线使用指南

本文说明重构后的训练相关代码该如何使用。日常推荐从统一入口 `cs336_basics/cli.py` 开始，旧脚本仍然保留为兼容入口。

所有命令都遵循项目约定，使用 `uv run <python_file_path>` 执行。

## 快速开始

查看统一入口支持的子命令：

```sh
uv run cs336_basics/cli.py --help
```

推荐的主流程是：

```sh
# 1. 训练 tokenizer 并编码训练集
uv run cs336_basics/cli.py preprocess-tinystories \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinystoriesV2-GPT4-train.npy \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --vocab-size 10000 \
  --dtype uint16 \
  --train-config-out train_config.generated.json

# 2. 用同一个 tokenizer 编码验证集
uv run cs336_basics/cli.py encode-validation \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/TinyStoriesV2-GPT4-valid.npy \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --dtype uint16

# 3. 训练模型
uv run cs336_basics/cli.py train --config train_config.json

# 4. 评估 checkpoint
uv run cs336_basics/cli.py eval \
  --config train_config.json \
  --checkpoint data/checkpoint/train_checkpoint_step_40000.pt

# 5. 从 checkpoint 生成文本
uv run cs336_basics/cli.py generate \
  --config train_config.json \
  --checkpoint data/checkpoint/train_checkpoint_step_40000.pt \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --prompt "Where is Tom and Lily?"
```

如果只是 smoke test，不提供 `data.train_path` 时训练会使用合成 token 数据；正式训练应在配置里设置真实的 `.npy` 或二进制 token 数据路径。

## 统一 CLI

统一入口位于 `cs336_basics/cli.py`，包含五个子命令：

```sh
uv run cs336_basics/cli.py train
uv run cs336_basics/cli.py eval
uv run cs336_basics/cli.py generate
uv run cs336_basics/cli.py preprocess-tinystories
uv run cs336_basics/cli.py encode-validation
```

### train

训练默认读取 `train_config.json`：

```sh
uv run cs336_basics/cli.py train --config train_config.json
```

打印完整默认配置示例：

```sh
uv run cs336_basics/cli.py train --print-example-config
```

每次训练都会先解析一个 run name。若配置为 `null`，训练启动时会自动生成类似 `train-20260715-143012-a1b2c3` 的名字，避免连续训练互相覆盖产物：

```json
{
  "run": {
    "name": null
  }
}
```

也可以手动固定名字，方便复现实验或继续写入同一个 run 目录：

```json
{
  "run": {
    "name": "tinystories-debug"
  }
}
```

训练过程中会按 `checkpoint.save_interval` 保存 step-specific checkpoint。若配置为：

```json
{
  "run": {
    "name": "tinystories-debug"
  },
  "checkpoint": {
    "path": "data/checkpoint/train_checkpoint.pt",
    "save_interval": 1000
  }
}
```

则第 1000 步会写到：

```text
data/checkpoint/tinystories-debug/train_checkpoint_step_1000.pt
```

同一次训练的日志和训练监控图也会写入对应 run 子目录，例如 `log/tinystories-debug/train.log` 和 `log/tinystories-debug/loss_curve.png`。

训练支持 `checkpoint.resume_from` 继续训练：

```json
{
  "run": {
    "name": "tinystories-debug"
  },
  "checkpoint": {
    "path": "data/checkpoint/train_checkpoint.pt",
    "save_interval": 1000,
    "resume_from": "data/checkpoint/tinystories-debug/train_checkpoint_step_1000.pt"
  }
}
```

`checkpoint.resume_from` 是输入 checkpoint 路径，不会被 run name 自动改写。若希望继续训练的后续产物仍写入原 run 目录，请显式设置同一个 `run.name`。

### eval

评估 checkpoint 的验证集 loss：

```sh
uv run cs336_basics/cli.py eval \
  --config train_config.json \
  --checkpoint data/checkpoint/train_checkpoint_step_40000.pt
```

验证数据默认使用配置里的 `eval.valid_path`。也可以临时覆盖：

```sh
uv run cs336_basics/cli.py eval \
  --config train_config.json \
  --checkpoint data/checkpoint/train_checkpoint_step_40000.pt \
  --valid-path data/TinyStoriesV2-GPT4-valid.npy \
  --mode sampled \
  --num-batches 50 \
  --batch-size 32
```

`--mode` 支持：

- `sampled`：随机采样若干 batch，适合训练中快速观察。
- `full`：按完整 context window 顺序扫过验证集，适合最终评估。

### generate

生成入口不再硬编码 checkpoint、tokenizer 目录和 prompt，必须显式传入：

```sh
uv run cs336_basics/cli.py generate \
  --config train_config.json \
  --checkpoint data/checkpoint/train_checkpoint_step_40000.pt \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --prompt "Where is Tom and Lily?" \
  --max-new-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --eos-token-id 0
```

`--device` 可覆盖配置中的设备，例如：

```sh
uv run cs336_basics/cli.py generate \
  --config train_config.json \
  --checkpoint data/checkpoint/train_checkpoint_step_40000.pt \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --prompt "Once upon a time" \
  --device cpu
```

## 数据流水线

数据相关逻辑集中在 `cs336_basics/data_pipeline.py`，CLI 中对应两个子命令。

### preprocess-tinystories

这个命令会完成三件事：

1. 从输入文本训练 BPE tokenizer。
2. 将输入文本编码为 1D `.npy` token 数据集。
3. 保存 tokenizer artifact，可选写出一份训练配置。

```sh
uv run cs336_basics/cli.py preprocess-tinystories \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinystoriesV2-GPT4-train.npy \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --vocab-size 10000 \
  --dtype uint16 \
  --train-config-out train_config.generated.json
```

输出的 tokenizer 目录包含：

```text
vocab.base64.json
merges.base64.json
metadata.json
```

### encode-validation

验证集应该使用训练集同一个 tokenizer 编码：

```sh
uv run cs336_basics/cli.py encode-validation \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/TinyStoriesV2-GPT4-valid.npy \
  --tokenizer-dir data/tinystories_train_tokenizer \
  --dtype uint16
```

`--dtype` 必须能容纳最大 token id。比如 `vocab_size=10000` 时 `uint16` 足够；更大的 vocab 可以直接使用默认的 `int64`。

## 配置文件

训练配置由 `cs336_basics/training/config.py` 中的 dataclass 定义。常用字段如下：

```json
{
  "model": {
    "vocab_size": 10000,
    "context_length": 256,
    "num_layers": 4,
    "d_model": 512,
    "num_heads": 16,
    "max_seq_len": 256,
    "theta": 10000.0,
    "d_ff": 1344
  },
  "data": {
    "train_path": "data/TinystoriesV2-GPT4-train.npy",
    "dtype": "int64",
    "use_memmap": true
  },
  "run": {
    "name": null
  },
  "eval": {
    "valid_path": "data/TinyStoriesV2-GPT4-valid.npy",
    "interval": 1000,
    "mode": "sampled",
    "num_batches": 50,
    "batch_size": null
  },
  "batch_size": 32,
  "total_steps": 40000,
  "device": "mps",
  "seed": 1150
}
```

设备字段支持：

- `auto`：优先 CUDA，其次 MPS，最后 CPU。
- `cpu`、`mps`、`cuda` 或其他 PyTorch device string。

训练监控图由 `plot` 配置控制。启用后会持续刷新一张 PNG 仪表盘：上半部分显示 raw train loss、平滑后的 train loss trend、validation loss 和 best validation 点；下半部分显示 learning rate，并在底部展示最新 train/val loss、step/s、elapsed/ETA 与总进度。若将 `plot.show` 设为 `true`，训练时还会打开一张 Matplotlib 实时窗口，并按同一刷新节奏更新窗口和 `plot.path` 指向的 PNG 文件。

```json
{
  "plot": {
    "enabled": true,
    "path": "log/loss_curve.png",
    "interval": 100,
    "width": 1000,
    "height": 720,
    "dpi": 120,
    "show": false,
    "pause_seconds": 0.001
  }
}
```

`plot.show=true` 需要当前 Python/Matplotlib 能使用交互式图形后端；在无桌面会话或只配置了 `Agg` 后端的环境里，建议保持默认的 `false`。

日志文件由 `logging.log_file` 控制，默认同时输出到 stdout。

## 代码结构

重构后，核心训练代码按职责拆到 `cs336_basics/training/`：

```text
cs336_basics/training/config.py           # 配置 dataclass 和 load_config
cs336_basics/training/runtime.py          # logging、device、seed
cs336_basics/training/factory.py          # build_model、build_optimizer
cs336_basics/training/data.py             # token dataset 加载
cs336_basics/training/checkpoint_eval.py  # checkpoint 验证 loss
cs336_basics/training/plotting.py         # loss curve 渲染
cs336_basics/training/trainer.py          # Trainer 和训练循环
```

数据预处理集中在：

```text
cs336_basics/data_pipeline.py
```

统一 CLI 在：

```text
cs336_basics/cli.py
```

## 旧入口兼容

以下文件仍然可以运行或 import，但推荐新代码使用统一 CLI 和 `cs336_basics/training/` 模块：

```sh
uv run cs336_basics/train_model.py --config train_config.json
uv run cs336_basics/validation_loss.py --config train_config.json --checkpoint <checkpoint.pt>
uv run cs336_basics/generate.py --config train_config.json --checkpoint <checkpoint.pt> --tokenizer-dir <tokenizer_dir> --prompt "..."
uv run cs336_basics/preprocess_tinystories.py ...
uv run cs336_basics/encode_validation.py ...
```

`cs336_basics/train_model.py` 继续导出旧测试和脚本常用的符号，例如：

```python
from cs336_basics.train_model import TrainingConfig, build_model, build_optimizer, train, evaluate_checkpoint
```

新代码更推荐直接从职责模块导入：

```python
from cs336_basics.training.config import TrainingConfig, load_config
from cs336_basics.training.trainer import Trainer, train
from cs336_basics.training.checkpoint_eval import evaluate_checkpoint
```

## 验证命令

重构后建议先跑训练相关 targeted tests：

```sh
uv run pytest unit_tests/test_train_model.py unit_tests/test_preprocess_tinystories.py unit_tests/test_encode_validation.py unit_tests/test_decode.py unit_tests/test_cli.py
```

最终跑全量测试：

```sh
uv run pytest
```

如果只想检查本次训练重构相关文件的 lint：

```sh
uv run ruff check \
  cs336_basics/cli.py \
  cs336_basics/data_pipeline.py \
  cs336_basics/training \
  cs336_basics/train_model.py \
  cs336_basics/validation_loss.py \
  cs336_basics/encode_validation.py \
  cs336_basics/preprocess_tinystories.py \
  cs336_basics/generate.py \
  unit_tests/test_cli.py
```

## 常见问题

### eval 报缺少 validation path

需要在配置中设置：

```json
{
  "eval": {
    "valid_path": "data/TinyStoriesV2-GPT4-valid.npy"
  }
}
```

或在命令行传入：

```sh
uv run cs336_basics/cli.py eval --config train_config.json --checkpoint <checkpoint.pt> --valid-path <valid.npy>
```

### generate 无法找到 tokenizer artifact

确认 `--tokenizer-dir` 指向包含以下三个文件的目录：

```text
vocab.base64.json
merges.base64.json
metadata.json
```

### checkpoint 文件名和配置里的 path 不完全一样

训练会先插入 run name 目录，再把 step 插入文件名。例如配置中 `checkpoint.path` 是：

```text
data/checkpoint/train_checkpoint.pt
```

第 40000 步的实际输出是：

```text
data/checkpoint/<run_name>/train_checkpoint_step_40000.pt
```

### 正式训练不想用合成数据

在配置中设置 `data.train_path`。如果没有设置，训练会使用一小段 synthetic token 数据，只适合 smoke test。
