"""
日志配置模块
支持 loguru 和标准 logging 库
"""

import sys
import os
import logging
from datetime import datetime

# 尝试导入 loguru，如果失败则使用标准 logging
try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    logger = logging.getLogger("weibo_risk")


def ensure_dir(path: str) -> str:
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = None,
    rotation: str = "1 day",
    retention: str = "7 days"
):
    """
    配置日志系统

    Args:
        log_level: 日志级别
        log_dir: 日志目录
        log_file: 日志文件名
        rotation: 日志轮转周期
        retention: 日志保留时间
    """
    ensure_dir(log_dir)

    if log_file is None:
        log_file = f"app_{datetime.now().strftime('%Y%m%d')}.log"

    log_path = os.path.join(log_dir, log_file)

    if HAS_LOGURU:
        # 使用 loguru
        logger.remove()

        # 控制台输出
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )

        # 文件输出
        logger.add(
            log_path,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )

        logger.info(f"日志系统初始化完成 (loguru)，日志目录: {log_dir}")
    else:
        # 使用标准 logging
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # 清除已有处理器
        logger.handlers.clear()

        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"日志系统初始化完成 (logging)，日志目录: {log_dir}")


def init_logger_from_config(config_path: str = "config/config.yaml"):
    """
    从配置文件初始化日志

    Args:
        config_path: 配置文件路径
    """
    try:
        from src.utils.helpers import load_config
        config = load_config(config_path)

        system_config = config.get("system", {})
        log_level = system_config.get("log_level", "INFO")

        paths_config = config.get("paths", {})
        log_dir = paths_config.get("logs", "logs")

        setup_logger(log_level=log_level, log_dir=log_dir)
    except Exception as e:
        setup_logger()
        logger.warning(f"从配置文件初始化日志失败: {e}")


if __name__ == "__main__":
    # 测试
    setup_logger(log_level="DEBUG")

    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")

