"""
快速测试脚本 - 验证项目是否能正常运行
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("微博舆情风险感知系统 - 快速测试")
print("=" * 50)

# 测试1: 数据采集模块
print("\n[1] 测试数据采集模块...")
try:
    from src.data_collection.weibo_collector import MockWeiboDataGenerator
    generator = MockWeiboDataGenerator()
    df = generator.generate_mock_data(10)
    print(f"    ✓ 成功生成 {len(df)} 条模拟数据")
    print(f"    字段: {list(df.columns)[:5]}...")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试2: 文本清洗模块
print("\n[2] 测试文本清洗模块...")
try:
    from src.preprocessing.text_cleaner import TextCleaner
    cleaner = TextCleaner()
    text = "今天天气真好！#北京生活# @小明 https://example.com"
    cleaned = cleaner.clean(text)
    tokens = cleaner.tokenize(text)
    print(f"    ✓ 原文: {text}")
    print(f"    ✓ 清洗: {cleaned}")
    print(f"    ✓ 分词: {tokens[:5]}...")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试3: 情感分析模块
print("\n[3] 测试情感分析模块...")
try:
    from src.analysis.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer(model_type="snownlp")
    result = analyzer.analyze_sentiment("今天心情非常好，太开心了！")
    print(f"    ✓ 极性: {result.polarity}")
    print(f"    ✓ 得分: {result.polarity_score:.3f}")
    print(f"    ✓ 情绪: {result.emotion}")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试4: 风险评估模块
print("\n[4] 测试风险评估模块...")
try:
    import pandas as pd
    from src.analysis.risk_perception import RiskPerceptionAnalyzer

    analyzer = RiskPerceptionAnalyzer()
    test_data = pd.DataFrame({
        "content": [
            "疫情形势严峻，很担心",
            "经济下行压力大",
            "今天天气不错"
        ],
        "created_at": pd.date_range("2024-01-01", periods=3, freq="H"),
        "reposts_count": [100, 50, 10]
    })
    result = analyzer.analyze_risk(test_data)
    print(f"    ✓ 风险得分: {result.overall_score:.1f}")
    print(f"    ✓ 风险等级: {result.risk_level.value}")
except Exception as e:
    print(f"    ✗ 失败: {e}")

# 测试5: 特征提取模块
print("\n[5] 测试特征提取模块...")
try:
    from src.preprocessing.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()
    features = extractor.extract_text_statistics("这是一段测试文本，用于检验特征提取功能")
    print(f"    ✓ 提取特征: {list(features.keys())}")
except Exception as e:
    print(f"    ✗ 失败: {e}")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)

