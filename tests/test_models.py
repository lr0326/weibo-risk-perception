"""
测试模型模块
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.risk_perception import RiskPerceptionAnalyzer, RiskLevel
from src.models.clustering import PopulationClusteringModel


class TestSentimentAnalyzer:
    """测试情感分析器"""

    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        return SentimentAnalyzer(model_type="snownlp")

    def test_analyze_positive(self, analyzer):
        """测试正面情感分析"""
        text = "今天心情非常好，一切都很顺利"
        result = analyzer.analyze_sentiment(text)

        assert result.polarity in ["positive", "neutral", "negative"]
        assert -1 <= result.polarity_score <= 1
        assert 0 <= result.intensity <= 1

    def test_analyze_negative(self, analyzer):
        """测试负面情感分析"""
        text = "太糟糕了，这件事让我非常失望和愤怒"
        result = analyzer.analyze_sentiment(text)

        assert result.polarity_score < 0.5  # 偏负面

    def test_analyze_batch(self, analyzer):
        """测试批量分析"""
        texts = ["好开心", "很难过", "今天下雨了"]
        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        for result in results:
            assert hasattr(result, "polarity")
            assert hasattr(result, "polarity_score")

    def test_get_summary(self, analyzer):
        """测试获取摘要"""
        texts = ["好开心", "很难过", "今天下雨了", "太棒了", "很失望"]
        results = analyzer.analyze_batch(texts)
        summary = analyzer.get_sentiment_summary(results)

        assert "total_count" in summary
        assert summary["total_count"] == 5
        assert "polarity_distribution" in summary


class TestRiskPerceptionAnalyzer:
    """测试风险感知分析器"""

    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        return RiskPerceptionAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        return pd.DataFrame({
            "content": [
                "疫情形势严峻，很担心",
                "经济下行压力大",
                "今天天气不错",
                "这个政策让人恐惧",
                "很生气，太过分了"
            ],
            "created_at": pd.date_range("2024-01-01", periods=5, freq="H"),
            "reposts_count": [100, 50, 10, 200, 80]
        })

    def test_analyze_risk(self, analyzer, sample_data):
        """测试风险分析"""
        result = analyzer.analyze_risk(sample_data)

        assert 0 <= result.overall_score <= 100
        assert isinstance(result.risk_level, RiskLevel)
        assert len(result.dimension_scores) > 0
        assert result.trend in ["rising", "stable", "declining"]

    def test_risk_level_determination(self, analyzer):
        """测试风险等级判定"""
        assert analyzer._determine_risk_level(20) == RiskLevel.LOW
        assert analyzer._determine_risk_level(55) == RiskLevel.MEDIUM
        assert analyzer._determine_risk_level(75) == RiskLevel.HIGH
        assert analyzer._determine_risk_level(90) == RiskLevel.CRITICAL

    def test_empty_data(self, analyzer):
        """测试空数据处理"""
        result = analyzer.analyze_risk(pd.DataFrame())

        assert result.overall_score == 0
        assert result.risk_level == RiskLevel.LOW


class TestClusteringModel:
    """测试聚类模型"""

    @pytest.fixture
    def model(self):
        """创建模型实例"""
        return PopulationClusteringModel()

    @pytest.fixture
    def sample_features(self):
        """创建测试特征"""
        np.random.seed(42)
        # 生成3个明显的簇
        cluster1 = np.random.randn(30, 5) + np.array([3, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 5) + np.array([0, 3, 0, 0, 0])
        cluster3 = np.random.randn(30, 5) + np.array([0, 0, 3, 0, 0])
        return np.vstack([cluster1, cluster2, cluster3])

    def test_fit(self, model, sample_features):
        """测试聚类"""
        labels = model.fit(sample_features, n_clusters=3)

        assert len(labels) == 90
        assert len(set(labels)) == 3

    def test_find_optimal_clusters(self, model, sample_features):
        """测试最优聚类数选择"""
        result = model.find_optimal_clusters(sample_features, min_clusters=2, max_clusters=5)

        assert "optimal_k" in result
        assert 2 <= result["optimal_k"] <= 5

    def test_generate_profiles(self, model, sample_features):
        """测试生成聚类���像"""
        model.fit(sample_features, n_clusters=3)
        profiles = model.generate_cluster_profiles(
            sample_features,
            feature_names=["f1", "f2", "f3", "f4", "f5"]
        )

        assert len(profiles) == 3
        for profile in profiles:
            assert profile.size > 0
            assert 0 <= profile.percentage <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

