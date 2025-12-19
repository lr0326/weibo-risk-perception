"""
启动分析脚本
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RiskPerceptionPipeline
from src.utils.logger import setup_logger
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="舆情风险分析工具")
    parser.add_argument("--input", "-i", type=str, default=None, help="输入数据文件路径")
    parser.add_argument("--keyword", "-k", type=str, default="社会热点", help="搜索关键词(采集新数据时)")
    parser.add_argument("--count", "-c", type=int, default=100, help="数据量")
    parser.add_argument("--mock", action="store_true", help="使用模拟数据")
    parser.add_argument("--output", "-o", type=str, default="data/outputs", help="输出目录")
    parser.add_argument("--report", "-r", type=str, default="html", choices=["html", "markdown"], help="报告格式")

    args = parser.parse_args()

    # 初始化日志
    setup_logger()

    logger.info("="*50)
    logger.info("舆情风险分析工具")
    logger.info("="*50)

    # 初始化流水线
    pipeline = RiskPerceptionPipeline()

    if args.input:
        # 从文件加载数据
        import pandas as pd
        logger.info(f"从文件加载数据: {args.input}")
        df = pd.read_csv(args.input)
        pipeline.raw_data = df

        # 运行分析
        df = pipeline.preprocess_data(df)
        df = pipeline.analyze_sentiment(df)
        assessment = pipeline.assess_risk(df)
        pipeline.predict_trend(df, train=True)
        pipeline.cluster_population(df)

        # 汇总结果
        results = {
            "sample_size": len(df),
            "risk_level": assessment.risk_level.value,
            "risk_score": assessment.overall_score,
            "dimension_scores": assessment.dimension_scores,
            "trend": assessment.trend,
            "warnings": assessment.warnings,
            "recommendations": assessment.recommendations
        }
        pipeline.results["summary"] = results
    else:
        # 运行完整分析
        results = pipeline.run_full_analysis(
            keywords=args.keyword,
            count=args.count,
            pages=2,
            use_mock=args.mock or True
        )

    # 打印结果
    print("\n" + "="*50)
    print("分析结果摘要")
    print("="*50)

    if "summary" in pipeline.results:
        summary = pipeline.results["summary"]
        print(f"样本量: {summary.get('sample_size', 0)}")
        print(f"风险等级: {summary.get('risk_level', '-')}")
        print(f"风险得分: {summary.get('risk_score', 0):.1f}")

        print("\n预警信息:")
        for warning in summary.get('warnings', []):
            print(f"  {warning}")

    # 生成报告
    report_path = pipeline.generate_report(format=args.report)
    print(f"\n报告已生成: {report_path}")

    # 保存数据
    pipeline.save_data(args.output)


if __name__ == "__main__":
    main()

