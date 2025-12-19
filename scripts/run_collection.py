"""
启动数据采集脚本
"""

import argparse
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.weibo_collector import WeiboDataCollector, MockWeiboDataGenerator
from src.utils.logger import setup_logger
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="微博数据采集工具")
    parser.add_argument("--keyword", "-k", type=str, default="社会热点", help="搜索关键词")
    parser.add_argument("--count", "-c", type=int, default=100, help="每页数量")
    parser.add_argument("--pages", "-p", type=int, default=10, help="采集页数")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--token", "-t", type=str, default=None, help="API访问令牌")

    args = parser.parse_args()

    # 初始化日志
    setup_logger()

    logger.info("="*50)
    logger.info("微博数据采集工具")
    logger.info("="*50)

    if args.mock:
        logger.info("使用模拟数据生成器")
        generator = MockWeiboDataGenerator()
        df = generator.generate_mock_data(args.count * args.pages)
    else:
        if not args.token:
            logger.error("请提供API访问令牌 (--token)")
            return

        collector = WeiboDataCollector(access_token=args.token)
        df = collector.search_weibo_by_keyword(
            keyword=args.keyword,
            count=args.count,
            pages=args.pages
        )

    if df.empty:
        logger.error("未采集到数据")
        return

    # 保存数据
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/weibo_data_{timestamp}.csv"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    logger.info(f"采集完成: {len(df)} 条数据")
    logger.info(f"数据已保存至: {args.output}")


if __name__ == "__main__":
    main()

