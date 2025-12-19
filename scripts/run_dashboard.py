"""
启动可视化仪表盘脚本
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.dashboard import Dashboard
from src.utils.logger import setup_logger
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="启动可视化仪表盘")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", "-p", type=int, default=8050, help="端口号")
    parser.add_argument("--debug", "-d", action="store_true", help="调试模式")
    parser.add_argument("--data", type=str, default=None, help="加载的数据文件")

    args = parser.parse_args()

    # 初始化日志
    setup_logger()

    logger.info("="*50)
    logger.info("启动可视化仪表盘")
    logger.info("="*50)

    # 初始化仪表盘
    dashboard = Dashboard()

    # 加载数据
    if args.data:
        import pandas as pd
        logger.info(f"加载数据: {args.data}")
        df = pd.read_csv(args.data)
        dashboard.update_data(df)

    logger.info(f"仪表盘地址: http://{args.host}:{args.port}")

    # 启动
    dashboard.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

