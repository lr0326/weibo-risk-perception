"""
基础使用示例
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_data_collection():
    """数据采集示例"""
    print("="*50)
    print("示例1: 数据采集")
    print("="*50)

    from src.data_collection.weibo_collector import MockWeiboDataGenerator

    # 使用模拟数据生成器
    generator = MockWeiboDataGenerator()
    df = generator.generate_mock_data(100)

    print(f"生成 {len(df)} 条模拟微博数据")
    print(f"字段: {list(df.columns)}")
    print(f"\n示例数据:")
    print(df.head())


def example_sentiment_analysis():
    """情感分析示例"""
    print("\n" + "="*50)
    print("示例2: 情感分析")
    print("="*50)

    from src.analysis.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer(model_type="snownlp")

    texts = [
        "今天心情真好，天气也很棒！",
        "这个政策太让人失望了",
        "明天要开会讨论项目进展",
        "疫情形势严峻，大家都很担心",
        "非常满意这次的服务体验"
    ]

    for text in texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\n文本: {text}")
        print(f"  极性: {result.polarity} (得分: {result.polarity_score:.3f})")
        print(f"  情绪: {result.emotion}")
        print(f"  强度: {result.intensity:.3f}")


def example_risk_assessment():
    """风险评估示例"""
    print("\n" + "="*50)
    print("示例3: 风险评估")
    print("="*50)

    import pandas as pd
    from src.analysis.risk_perception import RiskPerceptionAnalyzer

    analyzer = RiskPerceptionAnalyzer()

    # 创建测试数据
    test_data = pd.DataFrame({
        "content": [
            "疫情形势严峻，大家一定要做好防护",
            "经济下行压力大，很多企业面临困难",
            "对这个政策很担心，不知道会有什么影响",
            "今天天气不错，出去逛了逛街",
            "太生气了，这种事情怎么能发生",
            "病毒传播速度太快了，好害怕"
        ],
        "created_at": pd.date_range("2024-01-01", periods=6, freq="H"),
        "reposts_count": [100, 50, 200, 10, 80, 150]
    })

    # 评估风险
    result = analyzer.analyze_risk(test_data)

    print(f"综合风险得分: {result.overall_score:.1f}")
    print(f"风险等级: {result.risk_level.value}")
    print(f"趋势: {result.trend}")

    print("\n维度得分:")
    for dim, score in result.dimension_scores.items():
        print(f"  {dim}: {score:.1f}")

    print("\n预警信息:")
    for warning in result.warnings:
        print(f"  {warning}")


def example_text_processing():
    """文本处理示例"""
    print("\n" + "="*50)
    print("示例4: 文本处理")
    print("="*50)

    from src.preprocessing.text_cleaner import TextCleaner

    cleaner = TextCleaner()

    texts = [
        "今天天气真好！#北京生活# @小明 https://example.com [开心]",
        "疫情防控政策调整了，大家怎么看？🤔",
        "转发微博：经济形势分析..."
    ]

    for text in texts:
        cleaned = cleaner.clean(text)
        tokens = cleaner.tokenize(text)

        print(f"\n原文: {text}")
        print(f"清洗: {cleaned}")
        print(f"分词: {' / '.join(tokens)}")


def example_full_pipeline():
    """完整流程示例"""
    print("\n" + "="*50)
    print("示例5: 完整分析流程")
    print("="*50)

    from src.pipeline import RiskPerceptionPipeline

    # 初始化流水线
    pipeline = RiskPerceptionPipeline()

    # 运行完整分析（使用模拟数据）
    results = pipeline.run_full_analysis(
        keywords="社会热点",
        count=30,
        pages=1,
        use_mock=True
    )

    print("\n分析结果:")
    print(f"  样本量: {results.get('sample_size', 0)}")
    print(f"  风险等级: {results.get('risk_level', '-')}")
    print(f"  风险得分: {results.get('risk_score', 0):.1f}")


def main():
    """运行所有示例"""
    print("微博舆情风险感知系统 - 使用示例\n")

    example_data_collection()
    example_sentiment_analysis()
    example_risk_assessment()
    example_text_processing()
    example_full_pipeline()

    print("\n" + "="*50)
    print("所有示例运行完成!")
    print("="*50)


if __name__ == "__main__":
    main()

