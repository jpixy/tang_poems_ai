import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PoetryModel
from utils import PoetryDataset
from common import setup_logger
from config import CONFIG
import os
import time
import psutil
import gc

# 针对Kaggle的特殊处理
if "KAGGLE_URL_BASE" in os.environ:
    from tqdm.auto import tqdm  # 使用更适合notebook的版本

    tqdm._instances.clear()  # 清除现有进度条实例
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁用TensorFlow冗余日志
else:
    from tqdm import tqdm

logger = setup_logger("PoetryTrainer")


def print_training_config():
    logger.info("=" * 60)
    logger.info("训练配置详情:")
    logger.info(f"设备: {CONFIG.get('Device', 'use_gpu', fallback='auto')}")
    logger.info(f"Epochs: {CONFIG.getint('Training', 'epochs')}")
    logger.info(f"Batch size: {CONFIG.getint('Training', 'batch_size')}")
    logger.info(f"学习率: {CONFIG.getfloat('Training', 'learning_rate')}")
    logger.info("-" * 60)


def monitor_resources():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    if "KAGGLE_URL_BASE" in os.environ:
        logger.info(f"资源使用: CPU {cpu}% | 内存 {mem.percent}%")
    else:
        logger.info(
            f"系统资源: CPU {cpu}% | 内存 {mem.percent}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"
        )


def train():
    print_training_config()
    monitor_resources()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 模型初始化
    model_start = time.time()
    model = PoetryModel().to(device)
    logger.info(f"模型初始化完成，耗时: {time.time() - model_start:.2f}s")

    # 数据加载
    dataset = PoetryDataset(
        model.tokenizer,
        CONFIG["Paths"]["data_dir"],
        max_samples=CONFIG.getint("Training", "max_samples", fallback=500),
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.getint("Training", "batch_size"),
        shuffle=True,
        num_workers=min(4, os.cpu_count() // 2)
        if not "KAGGLE_URL_BASE" in os.environ
        else 0,
    )

    # 优化器
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=CONFIG.getfloat("Training", "learning_rate"),
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # 训练循环
    best_loss = float("inf")
    for epoch in range(CONFIG.getint("Training", "epochs")):
        epoch_start = time.time()
        model.model.train()
        total_loss = 0

        # Kaggle环境下使用优化的进度条
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}",
            leave=False if "KAGGLE_URL_BASE" in os.environ else True,
        )

        for batch in progress:
            inputs = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            if not "KAGGLE_URL_BASE" in os.environ:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch + 1}/{CONFIG.getint('Training', 'epochs')} | "
            f"Loss: {avg_loss:.4f} | "
            f"耗时: {epoch_time:.1f}s"
        )

        # 模型保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.model.state_dict(),
                os.path.join(CONFIG["Paths"]["model_save_dir"], "best_model.pth"),
            )
            logger.info(f"保存最佳模型 (Loss: {best_loss:.4f})")

    logger.info(f"训练完成! 最佳Loss: {best_loss:.4f}")


if __name__ == "__main__":
    os.makedirs(CONFIG["Paths"]["model_save_dir"], exist_ok=True)
    os.makedirs(CONFIG["Paths"]["log_dir"], exist_ok=True)

    try:
        train_start = time.time()
        train()
        logger.info(f"总运行时间: {(time.time() - train_start) / 60:.1f}分钟")
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise

