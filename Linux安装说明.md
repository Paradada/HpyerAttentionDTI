# HyperAttentionDTI — Linux 环境安装说明（Miniconda）

本文档说明如何在一台新的 Linux 机器（租卡）上，用 Miniconda 管理虚拟环境并运行 HyperAttentionDTI。
依赖已固化为 [requirements.txt](requirements.txt)，通过 conda 环境内的 `pip` 安装。

## 0. 前提条件（务必先确认）

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Linux | 命令以 `python3` / `conda` 为例 |
| Miniconda | 已安装（或见第 1 节安装） | 用于创建和管理虚拟环境 |
| Python | 3.10 – 3.12 | README 里的 3.6 已停止维护；下面用 conda 指定 3.12 |
| GPU | NVIDIA RTX 3080 Ti（12GB，Ampere `sm_86`） | **必须**。代码硬编码 `.cuda()`，无 GPU 无法运行 |
| NVIDIA 驱动 | 支持 CUDA 13.0 | `requirements.txt` 锁定 `torch 2.12.1+cu130`；租卡平台 `nvidia-smi` 显示 CUDA 13.0 即满足 |
| 显存 | 12GB | 可用 README 默认的 `Batch_size=32`（约 3.9GB 峰值显存，见 §6 说明） |

检查 GPU 和 CUDA 版本：

```bash
nvidia-smi
```

能打印出 RTX 3080 Ti 和 `CUDA Version: 13.0`（或更高）即说明驱动正常。

## 1. 安装 Miniconda（若尚未安装）

```bash
# 下载官方安装脚本（x86_64）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 执行安装，按提示操作（默认一路回车，最后 yes 允许 conda init）
bash Miniconda3-latest-Linux-x86_64.sh
```

安装完成后，重新打开终端（或 `source ~/.bashrc`）使 `conda` 命令生效，验证：

```bash
conda --version
```

## 2. 创建 conda 环境

```bash
conda create -n hyperattentiondti python=3.12 -y
```

- `-n hyperattentiondti`：环境名，可自定义。
- `python=3.12`：与已验证环境一致（3.10–3.12 均可）。

## 3. 激活环境

```bash
conda activate hyperattentiondti
```

激活成功后提示符前会出现 `(hyperattentiondti)`。

## 4. 安装依赖

进入包含 `requirements.txt` 的代码目录后执行：

```bash
cd /path/to/HpyerAttentionDTI/HpyerAttentionDTI
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 中已通过 `--extra-index-url` 指定 PyTorch 官方源，会自动安装 CUDA 版 `torch`（`2.12.1+cu130`）。

## 5. 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期输出类似 `2.12.1+cu130 True`。**`cuda.is_available()` 必须为 `True`**，否则训练会在第一处 `.cuda()` 报错。

## 6. 运行

```bash
python HyperAttentionDTI_main.py
```

数据文件 `data/Davis.txt`、`data/DrugBank.txt`、`data/KIBA.txt` 已随仓库提供，无需额外下载。

**关于显存与 batch size**：模型里的 Drug–Protein 注意力矩阵 `[B, 85, 979, 160]` 随 batch 增长很快。12GB 显存下 `Batch_size=32`（README 默认，峰值约 3.9GB）没问题；想更快可试 `64`。若换回 4GB 小显存机器，务必降到 `Batch_size=8`，否则会溢出到共享内存导致速度陡降（详见 [预实验记录与代码说明.md](预实验记录与代码说明.md)）。

## 7. 训练提速建议

**先澄清一个常见误区**：`Epoch` 只是「训练多少轮」，不影响单轮速度。把 `Epoch` 从 3 改成 200 不会变快，反而让总时长变成 200 倍（200 epoch × 5 折 ≈ 数百小时）。真正决定快慢的是下面的配置。

| 配置项 | 位置 | 默认/当前 | 12GB 3080 Ti 建议 | 说明 |
|---|---|---|---|---|
| `Batch_size` | hyperparameter.py:13 | 4（为 4GB 小显存设） | **32**（想更快试 64） | 步数减少 8×，最大收益，约 5–8× |
| `num_workers` | main.py:177 / 179 / 181 | 0 | **8** | SMILES/蛋白序列预处理并行 |
| `DATASET` | main.py:134 | KIBA（最大） | 验证阶段用 **Davis** | 数据量少约 5× |
| `K_Fold` | main.py:163 | 5 | 验证阶段用 **1** | 只跑一折先走通 |
| `Patience` | hyperparameter.py:15 | 50 | 保持 50 | 早停提前收敛，实际跑不满 Epoch |

**参考耗时（KIBA）**：`Batch_size=4` 约 21 分钟/epoch；改成 `32` 后约 4–5 分钟/epoch。

> 若还想更快，可加混合精度训练（AMP），Ampere 上通常还有 1.5–2× 提速，但需要改训练循环几行代码。

## 8. 若 CUDA / GPU 与默认不符

`requirements.txt` 默认锁定 `torch 2.12.1+cu130`，适配 **RTX 3080 Ti（Ampere）+ CUDA 13.0 驱动**。如果你的租卡属于以下情况，需要调整：

- **更老的卡（Pascal / Maxwell / Volta，如 GTX 10 系、V100）**：CUDA 13.0 已不再支持这些架构（最低 Turing `sm_75`），需改用 `cu126` 及对应的旧版 torch。
- **更新的卡（RTX 50 系 Blackwell `sm_120`）**：`cu130` 本身支持，一般无需改；若报错再升级 torch 小版本。
- **驱动较老（不支持 CUDA 13.0）**：改回 `cu128` 或 `cu126` 对应的 torch 版本。
- **平台只提供 torch 2.8.0（没有 2.12.1）**：torch 2.8.0 没有 cu130 构建，需把索引改成 `https://download.pytorch.org/whl/cu129` 并写 `torch==2.8.0+cu129`；cu129 仍可在 CUDA 13.0 驱动上运行（驱动向下兼容）。

修改时需同时改两处：

1. `requirements.txt` 顶部的 `--extra-index-url https://download.pytorch.org/whl/cuXXX`
2. `requirements.txt` 中 `torch==x.y.z+cuXXX` 的版本号

改完后重复第 4、5 步即可。

## 9. 备注

- 依赖清单基于 Windows 11 + Python 3.12.6 的已验证组合整理，目标租卡为 RTX 3080 Ti 12GB（Ampere）；预实验记录见 [预实验记录与代码说明.md](预实验记录与代码说明.md)。
- 若在 Linux 上遇到与 Windows 不同的报错，优先检查：GPU 是否可见（`nvidia-smi`）、驱动版本、以及 `torch.cuda.is_available()` 的结果。
