import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PoetryModel
from utils import PoetryDataset
from common import setup_logger
from config import CONFIG
from tqdm import tqdm
import os
import time
import psutil
import gc

logger = setup_logger("PoetryTrainer")


def print_training_config():
    logger.info("训练配置详情:")
    logger.info(f"设备: {CONFIG.get('Device', 'use_gpu', fallback='auto')}")
    logger.info(f"Epochs: {CONFIG.getint('Training', 'epochs')}")
    logger.info(f"Batch size: {CONFIG.getint('Training', 'batch_size')}")
    logger.info(f"学习率: {CONFIG.getfloat('Training', 'learning_rate')}")
    logger.info(f"梯度裁剪: {CONFIG.getfloat('Training', 'grad_clip')}")
    logger.info(f"权重裁剪: {CONFIG.getfloat('Training', 'weight_clip')}")
    logger.info(
        f"优化器: AdamW(betas={CONFIG['Optimizer']['betas']}, wd={CONFIG.getfloat('Optimizer', 'weight_decay')})"
    )


def monitor_resources():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    logger.info(
        f"系统资源: CPU {cpu}% | 内存 {mem.percent}% ({mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB)"
    )


def train():
    logger.info("=" * 60)
    logger.info("开始训练流程")
    logger.info("=" * 60)

    print_training_config()
    monitor_resources()

    # 设备选择
    use_gpu = CONFIG.get("Device", "use_gpu", fallback="auto")
    if use_gpu == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif use_gpu.lower() == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        logger.info("检测到CUDA设备")
        logger.info(f"设备名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(
            f"显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    else:
        logger.info("使用CPU进行训练")

    # 模型初始化
    model_start = time.time()
    model = PoetryModel().to(device)
    logger.info(f"模型初始化耗时: {time.time() - model_start:.2f}s")

    # 数据加载
    data_start = time.time()
    dataset = PoetryDataset(
        model.tokenizer,
        CONFIG["Paths"]["data_dir"],
        max_samples=CONFIG.getint("Training", "max_samples", fallback=500),
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.getint("Training", "batch_size"),
        shuffle=True,
        num_workers=min(4, os.cpu_count() // 2),
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    logger.info(f"数据加载耗时: {time.time() - data_start:.2f}s")
    logger.info(f"批量数据示例形状: {next(iter(loader))['input_ids'].shape}")

    # 优化器
    optimizer = optim.AdamW(
        [p for p in model.model.parameters() if p.requires_grad],
        lr=CONFIG.getfloat("Training", "learning_rate"),
        betas=tuple(map(float, CONFIG["Optimizer"]["betas"].split(","))),
        weight_decay=CONFIG.getfloat("Optimizer", "weight_decay"),
    )

    # 训练循环
    logger.info("=" * 60)
    logger.info("开始训练循环")
    logger.info("=" * 60)

    best_loss = float("inf")
    for epoch in range(CONFIG.getint("Training", "epochs")):
        epoch_start = time.time()
        model.model.train()
        total_loss = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress):
            iter_start = time.time()

            inputs = batch["input_ids"].to(device, non_blocking=True)
            masks = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model.model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if CONFIG.getfloat("Training", "grad_clip") > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.model.parameters(), CONFIG.getfloat("Training", "grad_clip")
                )

            optimizer.step()

            total_loss += loss.item()
            iter_time = time.time() - iter_start

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                iter_t=f"{iter_time:.2f}s",
                mem=f"{psutil.virtual_memory().percent}%",
            )

            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Epoch统计
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)

        logger.info("-" * 60)
        logger.info(f"Epoch {epoch + 1} 完成")
        logger.info(f"平均Loss: {avg_loss:.4f}")
        logger.info(f"耗时: {epoch_time:.2f}s ({epoch_time / len(loader):.2f}s/batch)")
        logger.info(
            f"预估剩余时间: {epoch_time * (CONFIG.getint('Training', 'epochs') - epoch - 1) / 60:.1f} min"
        )
        monitor_resources()

        # 模型保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.model.state_dict(),
                os.path.join(CONFIG["Paths"]["model_save_dir"], "best_model.pth"),
            )
            logger.info(f"保存最佳模型 (Loss: {best_loss:.4f})")

    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳Loss: {best_loss:.4f}")
    logger.info(f"总耗时: {(time.time() - model_start) / 60:.1f} 分钟")
    logger.info("=" * 60)


if __name__ == "__main__":
    os.makedirs(CONFIG["Paths"]["model_save_dir"], exist_ok=True)
    os.makedirs(CONFIG["Paths"]["log_dir"], exist_ok=True)

    try:
        train_start = time.time()
        train()
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise
    finally:
        logger.info(f"总运行时间: {(time.time() - train_start) / 60:.1f} 分钟")

