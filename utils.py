import os
import json
import logging
from torch.utils.data import Dataset
from config import CONFIG
import psutil

logger = logging.getLogger(__name__)


class PoetryDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_samples=None):
        self._print_data_info(data_dir)

        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_samples = max_samples or CONFIG.getint(
            "Training", "max_samples", fallback=500
        )

        logger.info("数据加载中...")
        self.poems = self._load_poems()
        self.data = self._preprocess_data()

        logger.info("数据集统计:")
        logger.info(f"总诗歌数: {len(self.poems)}")
        logger.info(f"实际使用样本: {len(self.data)}")
        logger.info(f"最大序列长度: {CONFIG.getint('Model', 'max_length')}")

        mem = psutil.virtual_memory()
        logger.info("内存使用情况:")
        logger.info(f"已使用: {mem.used / (1024**3):.2f} GB")
        logger.info(f"可用: {mem.available / (1024**3):.2f} GB")

    def _print_data_info(self, data_dir):
        logger.info("数据目录分析:")
        if not os.path.exists(data_dir):
            logger.error(f"目录不存在: {data_dir}")
            raise FileNotFoundError(data_dir)

        files = os.listdir(data_dir)
        json_files = [f for f in files if f.endswith(".json")]

        logger.info(f"总文件数: {len(files)}")
        logger.info(f"JSON文件数: {len(json_files)}")

        if len(json_files) > 0:
            sample_file = os.path.join(data_dir, json_files[0])
            with open(sample_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
            logger.info(f"示例文件结构: {type(sample)}")
            if isinstance(sample, dict):
                logger.info(f"示例字段: {list(sample.keys())}")

    def _load_poems(self):
        poems = []
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]

        logger.info(f"正在扫描 {len(json_files)} 个JSON文件...")
        processed_files = 0

        for filename in json_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        poems.extend(
                            [
                                p
                                for item in data
                                if isinstance(item, dict)
                                for p in item.get("paragraphs", [])
                                if isinstance(p, str)
                            ]
                        )
                    elif isinstance(data, dict):
                        poems.extend(
                            p for p in data.get("paragraphs", []) if isinstance(p, str)
                        )

                processed_files += 1
                if processed_files % 10 == 0:
                    logger.debug(f"已处理 {processed_files}/{len(json_files)} 个文件")

                if len(poems) >= self.max_samples:
                    break

            except Exception as e:
                logger.error(f"文件 {filename} 加载失败: {str(e)}")

        return [p for p in poems if len(p) >= 4][: self.max_samples]

    def _preprocess_data(self):
        processed = []
        max_len = CONFIG.getint("Model", "max_length")
        logger.info(f"预处理数据 (max_len={max_len})...")

        for i, poem in enumerate(self.poems):
            try:
                encoded = self.tokenizer(
                    poem,
                    padding="max_length",
                    max_length=max_len,
                    truncation=True,
                    return_tensors="pt",
                )
                processed.append(
                    {
                        "input_ids": encoded["input_ids"].squeeze(0),
                        "attention_mask": encoded["attention_mask"].squeeze(0),
                        "labels": encoded["input_ids"].squeeze(0),
                    }
                )

                if (i + 1) % 100 == 0:
                    logger.debug(f"已预处理 {i + 1}/{len(self.poems)} 个样本")

            except Exception as e:
                logger.error(f"样本 {i} 预处理失败: {str(e)}")

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

