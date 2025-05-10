from transformers import GPT2LMHeadModel, BertTokenizer
from config import CONFIG
from common import setup_logger
import os
import torch
import psutil

logger = setup_logger("Model")


class PoetryModel:
    def __init__(self):
        self._print_system_info()
        self._print_config()

        self.pretrained_path = CONFIG["Paths"]["pretrained_model"]
        self.freeze_spec = CONFIG["Model"]["freeze_layers"]

        logger.info("模型初始化配置:")
        logger.info(f"预训练路径: {self.pretrained_path}")
        logger.info(f"冻结层配置: {self.freeze_spec}")

        os.makedirs(self.pretrained_path, exist_ok=True)

        mem_before = psutil.virtual_memory().used / (1024**3)
        logger.info(f"内存使用前: {mem_before:.2f} GB")

        logger.info("加载模型中...")
        self.tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_path, low_cpu_mem_usage=True
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            self.pretrained_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

        mem_after = psutil.virtual_memory().used / (1024**3)
        logger.info(f"内存使用后: {mem_after:.2f} GB")
        logger.info(f"内存增量: {mem_after - mem_before:.2f} GB")

        self._freeze_layers()
        self.print_trainable_parameters()
        logger.info("模型初始化完成")

    def _print_system_info(self):
        import platform

        logger.info("系统信息:")
        logger.info(f"操作系统: {platform.system()} {platform.release()}")
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info(f"CPU核心数: {os.cpu_count()}")
        logger.info(f"总内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    def _print_config(self):
        logger.info("运行配置:")
        for section in CONFIG.sections():
            for key, val in CONFIG[section].items():
                logger.info(f"{section}.{key}: {val}")

    def _freeze_layers(self):
        total_layers = len(self.model.transformer.h)
        logger.info(f"层冻结处理 (总层数: {total_layers})")

        if self.freeze_spec.lower() == "none":
            logger.info("未冻结任何层")
            return

        for param in self.model.parameters():
            param.requires_grad = True

        if self.freeze_spec.startswith("all,-"):
            layers_to_unfreeze = [
                int(x) for x in self.freeze_spec.split("all,-")[1].split(",")
            ]

            for param in self.model.parameters():
                param.requires_grad = False

            for layer_num in layers_to_unfreeze:
                if layer_num < 0:
                    layer_num = total_layers + layer_num
                if 0 <= layer_num < total_layers:
                    for param in self.model.transformer.h[layer_num].parameters():
                        param.requires_grad = True
                    logger.info(f"解冻层 #{layer_num}")

    def print_trainable_parameters(self):
        trainable = 0
        total = 0
        for name, param in self.model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        logger.info("模型参数统计:")
        logger.info(f"总参数: {total:,}")
        logger.info(f"可训练参数: {trainable:,}")
        logger.info(f"冻结参数: {total - trainable:,}")
        logger.info(f"可训练比例: {trainable / total:.2%}")

    def to(self, device):
        logger.info(f"转移模型到设备: {device}")
        self.model = self.model.to(device)
        return self

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

