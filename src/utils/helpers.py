"""
辅助函数模块
提供通用的工具函数
"""

import os
import time
import hashlib
import functools
import logging
from typing import Dict, Any, Callable, Optional
from datetime import datetime

import yaml

# 使用兼容的 logger
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("weibo_risk")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return get_default_config()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        "system": {
            "name": "微博舆情风险感知系统",
            "version": "1.0.0",
            "debug": False
        },
        "data_collection": {
            "weibo_api": {
                "base_url": "https://api.weibo.com/2/",
                "rate_limit": 150
            },
            "crawler": {
                "batch_size": 100,
                "max_pages": 50,
                "sleep_interval": 1.0
            }
        },
        "preprocessing": {
            "text_cleaning": {
                "remove_urls": True,
                "remove_mentions": True,
                "remove_hashtags": False,
                "remove_emojis": False,
                "min_length": 5
            },
            "tokenization": {
                "engine": "jieba"
            },
            "feature_extraction": {
                "tfidf": {
                    "max_features": 5000,
                    "ngram_range": [1, 2]
                }
            }
        },
        "models": {
            "sentiment": {
                "model_type": "snownlp"
            },
            "lstm": {
                "hidden_dim": 128,
                "num_layers": 2,
                "sequence_length": 24,
                "learning_rate": 0.001,
                "epochs": 100,
                "early_stopping_patience": 10
            },
            "risk_perception": {
                "dimensions": ["health_risk", "economic_risk", "social_risk", "political_risk"],
                "weights": {
                    "health_risk": 0.30,
                    "economic_risk": 0.25,
                    "social_risk": 0.25,
                    "political_risk": 0.20
                }
            },
            "clustering": {
                "algorithm": "kmeans",
                "n_clusters": 5
            }
        },
        "warning": {
            "thresholds": {
                "risk_score": {
                    "low": 30,
                    "medium": 50,
                    "high": 70,
                    "critical": 85
                }
            }
        },
        "visualization": {
            "dashboard": {
                "host": "0.0.0.0",
                "port": 8050,
                "debug": False,
                "update_interval": 300
            },
            "reports": {
                "output_format": "html"
            }
        },
        "paths": {
            "data": {
                "raw": "data/raw",
                "processed": "data/processed",
                "models": "data/models",
                "outputs": "data/outputs"
            },
            "logs": "logs"
        }
    }


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟秒数
        backoff: 延迟倍增因子
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} 执行失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} 最终失败: {e}")

            raise last_exception

        return wrapper
    return decorator


def ensure_dir(path: str) -> str:
    """
    确保目录存在

    Args:
        path: 目录路径

    Returns:
        路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.debug(f"创建目录: {path}")
    return path


def generate_id(text: str = None) -> str:
    """
    生成唯一ID

    Args:
        text: 可选的基础文本

    Returns:
        唯一ID字符串
    """
    if text:
        base = f"{text}_{datetime.now().timestamp()}"
    else:
        base = str(datetime.now().timestamp())

    return hashlib.md5(base.encode()).hexdigest()[:12]


def format_number(num: int) -> str:
    """
    格式化数字显示

    Args:
        num: 数字

    Returns:
        格式化后的字符串
    """
    if num >= 100000000:
        return f"{num / 100000000:.1f}亿"
    elif num >= 10000:
        return f"{num / 10000:.1f}万"
    elif num >= 1000:
        return f"{num / 1000:.1f}k"
    else:
        return str(num)


def calculate_time_diff(dt: datetime) -> str:
    """
    计算时间差的友好显示

    Args:
        dt: 时间

    Returns:
        友好的时间差描述
    """
    now = datetime.now()

    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)

    diff = now - dt

    seconds = diff.total_seconds()

    if seconds < 60:
        return "刚刚"
    elif seconds < 3600:
        return f"{int(seconds / 60)}分钟前"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}小时前"
    elif seconds < 604800:
        return f"{int(seconds / 86400)}天前"
    else:
        return dt.strftime("%Y-%m-%d")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        截断后的文本
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def batch_process(items: list, batch_size: int = 100, process_func: Callable = None):
    """
    批量处理生成器

    Args:
        items: 待处理项目列表
        batch_size: 批次大小
        process_func: 处理函数（可选）

    Yields:
        批次数据
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        if process_func:
            yield process_func(batch)
        else:
            yield batch


def merge_dicts(*dicts: Dict) -> Dict:
    """
    深度合并多个字典

    Args:
        *dicts: 要合并的字典

    Returns:
        合并后的字典
    """
    result = {}

    for d in dicts:
        if not d:
            continue

        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value

    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法

    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值

    Returns:
        结果
    """
    if denominator == 0:
        return default
    return numerator / denominator


class Timer:
    """计时器上下文管理器"""

    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name}耗时: {elapsed:.2f}秒")

    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


if __name__ == "__main__":
    # 测试
    print(f"格式化数字: {format_number(12345678)}")
    print(f"生成ID: {generate_id('test')}")
    print(f"截断文本: {truncate_text('这是一段很长的测试文本' * 10, 50)}")

    # 测试计时器
    with Timer("测试"):
        time.sleep(0.5)

