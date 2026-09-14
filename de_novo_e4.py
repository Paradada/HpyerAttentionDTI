# -*- coding: utf-8 -*-
"""
de_novo_e4.py —— 论文第 3.4 节「Performance evaluation under de novo setting」
实验 E4（novel drug + novel protein，药物与蛋白质均 unseen）。

作用：
    在不修改仓库任何现有代码的前提下，通过 import 100% 复用：
      - 模型架构   : model.AttentionDTI
      - 数据集解析 : dataset.CustomDataSet / dataset.collate_fn
      - 超参数     : hyperparameter.hyperparameter
      - early stop : pytorchtools.EarlyStopping
      - 指标计算   : HyperAttentionDTI_main.test_precess（准确率/精确率/召回率/AUC/AUPR）
    仅将「数据集划分逻辑」替换为 de novo E4 定义：
        随机抽取 20% 的独特药物与 20% 的独特蛋白质作为 unseen，
        所有（unseen 药物, unseen 蛋白质）两两交互构成测试集；
        剩余 DTI 按 4:1 随机划分为训练集 + 验证集。
    最终执行单次实验（不做 5 折、不做多次重复）。

用法：
    python de_novo_e4.py --dataset DrugBank --seed 42 --data_ratio 0.2
"""

import argparse
import os
import random
import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter

# ---- 100% 复用现有代码（不做任何修改）----
from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping
from HyperAttentionDTI_main import test_precess

SETTING = "E4"


# ======================================================================================
# 一、数据读取与 de novo E4 划分（本文件唯一新增逻辑）
# ======================================================================================

def load_lines(data_path):
    """读入原始数据文件，每行 5 列：Drug_ID Protein_ID SMILES Sequence Label。"""
    with open(data_path, "r") as f:
        return f.read().strip().split("\n")


def get_drug_id(line):
    """提取药物 ID（第 1 列）。"""
    return line.strip().split()[0]


def get_protein_id(line):
    """提取蛋白质 ID（第 2 列）。"""
    return line.strip().split()[1]


def split_de_novo_drug_protein(lines, ratio, seed):
    """E4 划分：随机 ratio 的独特药物与独特蛋白质为 unseen，二者两两交互作为测试集。

    测试集定义为「药物 unseen 且 蛋白质 unseen」的所有交互（novel drug + novel protein），
    其余交互（药物 seen 或 蛋白质 seen）作为训练/验证的候选。

    Args:
        lines (list[str]): 原始数据行列表。
        ratio (float): unseen 实体比例（论文为 0.2，药物与蛋白质各取 20%）。
        seed (int): 随机种子，保证划分可复现。

    Returns:
        tuple[list[str], list[str]]: (remain_lines, test_lines)。
    """
    rng = np.random.default_rng(seed)
    unique_drugs = sorted({get_drug_id(l) for l in lines})
    unique_proteins = sorted({get_protein_id(l) for l in lines})

    n_unseen_drugs = max(1, int(ratio * len(unique_drugs)))
    n_unseen_proteins = max(1, int(ratio * len(unique_proteins)))
    unseen_drugs = set(rng.choice(unique_drugs, size=n_unseen_drugs, replace=False).tolist())
    unseen_proteins = set(rng.choice(unique_proteins, size=n_unseen_proteins, replace=False).tolist())

    test_lines = [l for l in lines
                  if get_drug_id(l) in unseen_drugs and get_protein_id(l) in unseen_proteins]
    remain_lines = [l for l in lines
                    if not (get_drug_id(l) in unseen_drugs and get_protein_id(l) in unseen_proteins)]
    return remain_lines, test_lines


def train_valid_split(remain_lines, seed):
    """剩余 DTI 按 4:1 随机划分为训练集 + 验证集（与原始 random_split 语义一致）。"""
    dataset = CustomDataSet(remain_lines)
    n = len(dataset)
    valid_size = int(0.2 * n)  # 4:1 -> valid 占剩余集的 20%
    train_size = n - valid_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = torch.utils.data.random_split(
        dataset, [train_size, valid_size], generator=generator)
    return train_ds, valid_ds


# ======================================================================================
# 二、结果输出（单次实验，直接打印指标，不再做 mean±std）
# ======================================================================================

def print_and_save_results(dataset, setting, metrics, save_path):
    """把单次实验的指标打印到控制台，并写入 results.txt。"""
    lines = [
        "The {} de novo {} model's results (single run):".format(dataset, setting),
        "Loss:{:.5f}".format(metrics["Loss"]),
        "Accuracy:{:.4f}".format(metrics["Accuracy"]),
        "Precision:{:.4f}".format(metrics["Precision"]),
        "Recall:{:.4f}".format(metrics["Recall"]),
        "AUC:{:.4f}".format(metrics["AUC"]),
        "AUPR:{:.4f}".format(metrics["AUPR"]),
    ]
    with open(os.path.join(save_path, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)


# ======================================================================================
# 三、训练 + 评估引擎（与 HyperAttentionDTI_main.run_dataset 完全一致，仅划分不同）
# ======================================================================================

def run_experiment(args):
    """执行单次 de novo 实验。训练/验证/测试/优化器/损失/早停/指标全部复用现有逻辑。"""
    DATASET = args.dataset
    seed = args.seed
    ratio = args.data_ratio

    # 1) 固定随机种子（与原始入口一致）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2) 初始化超参数（100% 复用 hyperparameter 类），并支持命令行覆盖
    hp = hyperparameter()
    if args.batch_size is not None:
        hp.Batch_size = args.batch_size
    if args.epoch is not None:
        hp.Epoch = args.epoch
    if args.lr is not None:
        hp.Learning_rate = args.lr
    if args.patience is not None:
        hp.Patience = args.patience

    # 3) 读入数据 + 类别权重（与原始 run_dataset 一致）
    lines = load_lines("./data/{}.txt".format(DATASET))
    if DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).cuda()
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).cuda()
    else:  # DrugBank
        weight_CE = None

    # 4) de novo E4 划分（唯一与原始不同的地方）
    remain_lines, test_lines = split_de_novo_drug_protein(lines, ratio, seed)
    train_ds, valid_ds = train_valid_split(remain_lines, seed)
    test_ds = CustomDataSet(test_lines)

    print("=" * 60)
    print("Dataset: {} | de novo {} | ratio: {}".format(DATASET, SETTING, ratio))
    print("total={}, unseen_test={}, remaining={} -> train={}, valid={}".format(
        len(lines), len(test_lines), len(remain_lines), len(train_ds), len(valid_ds)))
    print("=" * 60)

    # 5) DataLoader（与原始一致；Windows 下建议 num_workers=0，原始代码为 8）
    train_loader = DataLoader(train_ds, batch_size=hp.Batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=hp.Batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=hp.Batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    # 6) 模型 + 权重初始化（与原始一致）
    model = AttentionDTI(hp).cuda()
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if "bias" in name:
            bias_p += [p]
        else:
            weight_p += [p]

    # 7) 优化器 + 学习率调度 + 损失函数（与原始一致）
    optimizer = optim.AdamW(
        [{"params": weight_p, "weight_decay": hp.weight_decay},
         {"params": bias_p, "weight_decay": 0}],
        lr=hp.Learning_rate)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
        cycle_momentum=False, step_size_up=len(train_ds) // hp.Batch_size)
    Loss = nn.CrossEntropyLoss(weight=weight_CE)

    # 8) 输出目录 + early stopping（与原始一致）
    save_path = "./{}/{}".format(DATASET, SETTING)
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=save_path)
    early_stopping = EarlyStopping(savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

    # 9) 训练循环（与原始 run_dataset 完全一致，仅单次而非 5 折）
    print("Training...")
    start = timeit.default_timer()
    for epoch in range(1, hp.Epoch + 1):
        # ---- train ----
        train_pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))
        train_losses = []
        model.train()
        for _, train_data in train_pbar:
            compounds, proteins, labels = train_data
            compounds, proteins, labels = compounds.cuda(), proteins.cuda(), labels.cuda()
            optimizer.zero_grad()
            predicted = model(compounds, proteins)
            loss = Loss(predicted, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_loss_a_epoch = np.average(train_losses)

        # ---- valid（100% 复用 test_precess 计算指标）----
        valid_pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
        _, _, valid_loss_a_epoch, Acc_dev, Prec_dev, Rec_dev, AUC_dev, PRC_dev = \
            test_precess(model, valid_pbar, Loss)

        # ---- 记录 tensorboard ----
        writer.add_scalar("Train Loss", train_loss_a_epoch, epoch)
        writer.add_scalar("Valid Loss", valid_loss_a_epoch, epoch)
        writer.add_scalar("Valid AUC", AUC_dev, epoch)
        writer.add_scalar("Valid AUPR", PRC_dev, epoch)
        writer.add_scalar("Valid Accuracy", Acc_dev, epoch)
        writer.add_scalar("Valid Precision", Prec_dev, epoch)
        writer.add_scalar("Valid Recall", Rec_dev, epoch)
        writer.add_scalar("Learn Rate", optimizer.param_groups[0]["lr"], epoch)

        epoch_len = len(str(hp.Epoch))
        print_msg = (f"[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] "
                     f"train_loss: {train_loss_a_epoch:.5f} "
                     f"valid_loss: {valid_loss_a_epoch:.5f} "
                     f"valid_AUC: {AUC_dev:.5f} "
                     f"valid_PRC: {PRC_dev:.5f} "
                     f"valid_Accuracy: {Acc_dev:.5f} "
                     f"valid_Precision: {Prec_dev:.5f} "
                     f"valid_Recall: {Rec_dev:.5f}")
        print(print_msg)

        early_stopping(valid_loss_a_epoch, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("Training finished in {:.1f}s".format(timeit.default_timer() - start))

    # 10) 加载验证集 loss 最低的 checkpoint 再做最终测试
    best_ckpt = os.path.join(save_path, "valid_best_checkpoint.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location="cuda"))
    model.eval()

    # 11) 测试集评估（100% 复用 test_precess 计算指标）
    test_pbar = tqdm(enumerate(BackgroundGenerator(test_loader)), total=len(test_loader))
    Y_test, P_test, test_loss, Acc_test, Prec_test, Rec_test, AUC_test, PRC_test = \
        test_precess(model, test_pbar, Loss)

    metrics = {
        "Loss": test_loss,
        "Accuracy": Acc_test,
        "Precision": Prec_test,
        "Recall": Rec_test,
        "AUC": AUC_test,
        "AUPR": PRC_test,
    }
    print_and_save_results(DATASET, SETTING, metrics, save_path)

    # 12) 保存测试集预测（真实标签 预测标签）
    pred_path = os.path.join(save_path, "{}_de_novo_{}_prediction.txt".format(DATASET, SETTING))
    with open(pred_path, "w") as f:
        for t, p in zip(Y_test, P_test):
            f.write("{} {}\n".format(t, p))

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="HyperAttentionDTI de novo E4 (novel drug + novel protein) experiment")
    parser.add_argument("--dataset", type=str, default="DrugBank",
                        choices=["DrugBank", "Davis", "KIBA"])
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_ratio", type=float, default=0.2, help="unseen 实体比例")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 hp.Batch_size")
    parser.add_argument("--epoch", type=int, default=None, help="覆盖 hp.Epoch")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 hp.Learning_rate")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 hp.Patience")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 进程数")
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("该脚本复用现有代码（硬编码 .cuda()），需要 GPU 环境。")
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
