"""
基于微博数据的社会风险感知与舆情预测系统

主入口文件
"""

import argparse
import sys

from src.pipeline import RiskPerceptionPipeline, main as run_pipeline
from src.utils.logger import setup_logger


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="微博舆情风险感知系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py analyze --keyword "社会热点" --mock
  python main.py dashboard --port 8050
  python main.py collect --keyword "疫情" --pages 5 --mock
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 分析命令
    analyze_parser = subparsers.add_parser("analyze", help="运行分析流程")
    analyze_parser.add_argument("--keyword", "-k", type=str, default="社会热点", help="搜索关键词")
    analyze_parser.add_argument("--count", "-c", type=int, default=100, help="数据量")
    analyze_parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    analyze_parser.add_argument("--output", "-o", type=str, default="data/outputs", help="输出目录")

    # 仪表盘命令
    dashboard_parser = subparsers.add_parser("dashboard", help="启动可视化仪表盘")
    dashboard_parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    dashboard_parser.add_argument("--port", "-p", type=int, default=8050, help="端口号")
    dashboard_parser.add_argument("--debug", "-d", action="store_true", help="调试模式")

    # 采集命令
    collect_parser = subparsers.add_parser("collect", help="数据采集")
    collect_parser.add_argument("--keyword", "-k", type=str, required=True, help="搜索关键词")
    collect_parser.add_argument("--pages", "-p", type=int, default=10, help="采集页数")
    collect_parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    collect_parser.add_argument("--output", "-o", type=str, default="data/raw", help="输出目录")

    args = parser.parse_args()

    if args.command is None:
        # 默认运行分析流程
        run_pipeline()
    elif args.command == "analyze":
        setup_logger()
        pipeline = RiskPerceptionPipeline()
        results = pipeline.run_full_analysis(
            keywords=args.keyword,
            count=args.count,
            pages=2,
            use_mock=args.mock or True
        )
        pipeline.generate_report()
        pipeline.save_data(args.output)

    elif args.command == "dashboard":
        from src.visualization.dashboard import Dashboard
        setup_logger()
        dashboard = Dashboard()
        dashboard.run(host=args.host, port=args.port, debug=args.debug)

    elif args.command == "collect":
        from src.data_collection.weibo_collector import MockWeiboDataGenerator
        import os
        from datetime import datetime

        setup_logger()

        if args.mock:
            generator = MockWeiboDataGenerator()
            df = generator.generate_mock_data(100 * args.pages)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"weibo_{timestamp}.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"数据已保存至: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

