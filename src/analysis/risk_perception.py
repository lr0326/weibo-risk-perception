"""
风险感知分析模块
多维度社会风险感知评估
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter

from sklearn.preprocessing import StandardScaler
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config
from src.analysis.sentiment_analyzer import SentimentAnalyzer


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskDimension(Enum):
    """风险维度"""
    HEALTH = "health_risk"
    ECONOMIC = "economic_risk"
    SOCIAL = "social_risk"
    POLITICAL = "political_risk"
    ENVIRONMENTAL = "environmental_risk"


@dataclass
class RiskAssessment:
    """风险评估结果"""
    overall_score: float  # 0-100
    risk_level: RiskLevel
    dimension_scores: Dict[str, float]
    key_factors: List[str]
    trend: str  # rising, stable, declining
    warnings: List[str]
    recommendations: List[str]


class RiskPerceptionAnalyzer:
    """
    风险感知分析器

    功能：
    - 多维度风险评估
    - 风险趋势分析
    - 预警��成
    - 群体风险感知建模
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化风险感知分析器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 风险配置
        risk_config = self.config.get("models", {}).get("risk_perception", {})
        self.dimensions = risk_config.get("dimensions", [
            "health_risk", "economic_risk", "social_risk", "political_risk"
        ])
        self.weights = risk_config.get("weights", {
            "health_risk": 0.30,
            "economic_risk": 0.25,
            "social_risk": 0.25,
            "political_risk": 0.20
        })

        # 预警阈值
        warning_config = self.config.get("warning", {}).get("thresholds", {})
        self.risk_thresholds = warning_config.get("risk_score", {
            "low": 30, "medium": 50, "high": 70, "critical": 85
        })

        # 情感分析器
        self.sentiment_analyzer = SentimentAnalyzer(model_type="snownlp", config_path=config_path)

        # 风险关键词
        self._load_risk_keywords()

        logger.info("风险感知分析器初始化完成")

    def _load_risk_keywords(self):
        """加载风险关键词"""
        self.risk_keywords = {
            "health_risk": {
                "high": ["疫情", "病毒", "感染", "死亡", "病例", "确诊", "传染", "爆发", "隔离", "封城"],
                "medium": ["病", "医院", "治疗", "症状", "健康", "防护", "口罩", "疫苗", "核酸"],
                "low": ["卫生", "保健", "锻炼", "养生"]
            },
            "economic_risk": {
                "high": ["破产", "倒闭", "失业", "裁员", "暴跌", "崩盘", "危机", "衰退"],
                "medium": ["下跌", "亏损", "困难", "压力", "通胀", "房价", "物价", "涨价"],
                "low": ["经济", "市场", "投资", "理财"]
            },
            "social_risk": {
                "high": ["暴力", "冲突", "骚乱", "示威", "抗议", "群体事件", "恐慌"],
                "medium": ["不满", "抱怨", "纠纷", "矛盾", "争议", "质疑"],
                "low": ["讨论", "关注", "建议", "呼吁"]
            },
            "political_risk": {
                "high": ["动荡", "政变", "制裁", "战争", "对抗"],
                "medium": ["紧张", "摩擦", "争端", "施压", "博弈"],
                "low": ["政策", "措施", "改革", "调整"]
            }
        }

        # 情绪-风险映射
        self.emotion_risk_weights = {
            "fear": 1.5,
            "anger": 1.3,
            "sadness": 1.1,
            "disgust": 1.2,
            "surprise": 0.8,
            "joy": 0.3,
            "neutral": 0.5
        }

    def analyze_risk(self, df: pd.DataFrame) -> RiskAssessment:
        """
        分析数据集的风险

        Args:
            df: 包含微博数据的DataFrame

        Returns:
            RiskAssessment对象
        """
        if df.empty:
            return self._empty_assessment()

        # 情感分析
        if "sentiment_score" not in df.columns:
            df = self.sentiment_analyzer.analyze_dataframe(df)

        # 计算各维度风险得分
        dimension_scores = {}
        for dimension in self.dimensions:
            score = self._calculate_dimension_score(df, dimension)
            dimension_scores[dimension] = score

        # 计算综合得分
        overall_score = sum(
            score * self.weights.get(dim, 0.25)
            for dim, score in dimension_scores.items()
        )

        # 确定风险等级
        risk_level = self._determine_risk_level(overall_score)

        # 识别关键因素
        key_factors = self._identify_key_factors(df, dimension_scores)

        # 分析趋势
        trend = self._analyze_trend(df)

        # 生成预警
        warnings = self._generate_warnings(overall_score, dimension_scores, df)

        # 生成建议
        recommendations = self._generate_recommendations(risk_level, dimension_scores)

        logger.info(f"风险评估完成 - 得分: {overall_score:.1f}, 等级: {risk_level.value}")

        return RiskAssessment(
            overall_score=overall_score,
            risk_level=risk_level,
            dimension_scores=dimension_scores,
            key_factors=key_factors,
            trend=trend,
            warnings=warnings,
            recommendations=recommendations
        )

    def _calculate_dimension_score(self, df: pd.DataFrame, dimension: str) -> float:
        """计算单个维度的风险得分"""
        keywords = self.risk_keywords.get(dimension, {})

        # 关键词匹配得分
        keyword_score = 0.0
        total_matches = 0

        for level, words in keywords.items():
            level_weight = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(level, 1.0)

            for text in df["content"].fillna(""):
                matches = sum(1 for word in words if word in text)
                keyword_score += matches * level_weight
                total_matches += matches

        # 归一化
        if len(df) > 0:
            keyword_score = min(50, (keyword_score / len(df)) * 10)

        # 情感得分（负面情感增加风险）
        sentiment_scores = df.get("sentiment_score", pd.Series([0]))
        avg_sentiment = sentiment_scores.mean()
        sentiment_risk = (1 - avg_sentiment) * 25  # 负面情感贡献0-25分

        # 情绪得分
        emotions = df.get("emotion", pd.Series(["neutral"]))
        emotion_risk = 0.0
        for emotion in emotions:
            weight = self.emotion_risk_weights.get(emotion, 0.5)
            emotion_risk += weight
        emotion_risk = min(25, (emotion_risk / max(len(df), 1)) * 15)

        # 综合得分
        total_score = keyword_score + sentiment_risk + emotion_risk

        return min(100, max(0, total_score))

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """确定风险等级"""
        if score >= self.risk_thresholds.get("critical", 85):
            return RiskLevel.CRITICAL
        elif score >= self.risk_thresholds.get("high", 70):
            return RiskLevel.HIGH
        elif score >= self.risk_thresholds.get("medium", 50):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _identify_key_factors(
        self,
        df: pd.DataFrame,
        dimension_scores: Dict[str, float]
    ) -> List[str]:
        """识别关键风险因素"""
        factors = []

        # 按维度得分排序
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for dim, score in sorted_dims[:2]:
            if score > 30:
                dim_name = {
                    "health_risk": "健康风险",
                    "economic_risk": "经济风险",
                    "social_risk": "社会风险",
                    "political_risk": "政治风险"
                }.get(dim, dim)
                factors.append(f"{dim_name}较高 (得分: {score:.1f})")

        # 负面情感比例
        if "sentiment_polarity" in df.columns:
            negative_ratio = (df["sentiment_polarity"] == "negative").mean()
            if negative_ratio > 0.3:
                factors.append(f"负面情绪占比较高 ({negative_ratio:.1%})")

        # 高传播内容
        if "reposts_count" in df.columns:
            high_spread = df[df["reposts_count"] > df["reposts_count"].quantile(0.9)]
            if len(high_spread) > 0:
                factors.append(f"存在{len(high_spread)}条高传播内容")

        return factors

    def _analyze_trend(self, df: pd.DataFrame) -> str:
        """分析风险趋势"""
        if "created_at" not in df.columns or len(df) < 10:
            return "stable"

        try:
            df = df.copy()
            df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.sort_values("created_at")

            # 分成前后两半
            mid_point = len(df) // 2
            first_half = df.iloc[:mid_point]
            second_half = df.iloc[mid_point:]

            # 比较负面情感比例
            if "sentiment_score" in df.columns:
                first_avg = first_half["sentiment_score"].mean()
                second_avg = second_half["sentiment_score"].mean()

                # 负面情感增加 = 风险上升
                if second_avg < first_avg - 0.1:
                    return "rising"
                elif second_avg > first_avg + 0.1:
                    return "declining"

            return "stable"

        except Exception as e:
            logger.debug(f"趋势分析失败: {e}")
            return "stable"

    def _generate_warnings(
        self,
        overall_score: float,
        dimension_scores: Dict[str, float],
        df: pd.DataFrame
    ) -> List[str]:
        """生成预警信息"""
        warnings = []

        # 综合风险预警
        if overall_score >= 85:
            warnings.append("⚠️ 【严重】综合风险指数超过临界值，需立即关注")
        elif overall_score >= 70:
            warnings.append("⚠️ 【警告】综合风险指数较高，建议密切监控")
        elif overall_score >= 50:
            warnings.append("⚠️ 【注意】综合风险指数中等，需要关注")

        # 维度预警
        for dim, score in dimension_scores.items():
            if score >= 70:
                dim_name = {
                    "health_risk": "健康",
                    "economic_risk": "经济",
                    "social_risk": "社会",
                    "political_risk": "政治"
                }.get(dim, dim)
                warnings.append(f"⚠️ {dim_name}风险维度得分较高 ({score:.1f})")

        # 情绪预警
        if "emotion" in df.columns:
            fear_count = (df["emotion"] == "fear").sum()
            anger_count = (df["emotion"] == "anger").sum()

            if fear_count / max(len(df), 1) > 0.2:
                warnings.append(f"⚠️ 恐惧情绪表达占比较高 ({fear_count}条)")
            if anger_count / max(len(df), 1) > 0.2:
                warnings.append(f"⚠️ 愤怒情绪表达占比较高 ({anger_count}条)")

        return warnings

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        dimension_scores: Dict[str, float]
    ) -> List[str]:
        """生成应对建议"""
        recommendations = []

        # 通用建议
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("建议立即启动应急响应机制")
            recommendations.append("建议增加舆情监测频率至实时监控")
            recommendations.append("建议准备官方回应声明")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("建议持续关注舆情发展态势")
            recommendations.append("建议准备风险应对预案")
        else:
            recommendations.append("当前风险可控，保持常规监测即可")

        # 维度针对性建议
        max_dim = max(dimension_scores, key=dimension_scores.get)
        if dimension_scores[max_dim] >= 50:
            if max_dim == "health_risk":
                recommendations.append("建议加强健康信息发布和科普宣传")
                recommendations.append("建议及时发布权威健康指导")
            elif max_dim == "economic_risk":
                recommendations.append("建议关注经济相关诉求和诉求")
                recommendations.append("建议加强经济政策解读")
            elif max_dim == "social_risk":
                recommendations.append("建议关注社会热点事件的舆论导向")
                recommendations.append("建议加强社会稳定相关信息发布")
            elif max_dim == "political_risk":
                recommendations.append("建议审慎评估政治敏感性")
                recommendations.append("建议加强正面舆论引导")

        return recommendations

    def _empty_assessment(self) -> RiskAssessment:
        """返回空评估结果"""
        return RiskAssessment(
            overall_score=0.0,
            risk_level=RiskLevel.LOW,
            dimension_scores={dim: 0.0 for dim in self.dimensions},
            key_factors=[],
            trend="stable",
            warnings=["数据为空，无法进行风险评估"],
            recommendations=["请确保有足够的数据进行分析"]
        )

    def calculate_risk_index(
        self,
        df: pd.DataFrame,
        time_window: str = "1H"
    ) -> pd.DataFrame:
        """
        计算时序风险指数

        Args:
            df: 输入数据
            time_window: 时间窗口

        Returns:
            风险指数时序数据
        """
        if "created_at" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.set_index("created_at")

        # 情感分析
        if "sentiment_score" not in df.columns:
            df = self.sentiment_analyzer.analyze_dataframe(df.reset_index())
            df = df.set_index("created_at")

        # 按时间窗口聚合
        risk_series = df.resample(time_window).agg({
            "sentiment_score": "mean",
            "content": "count"
        }).rename(columns={"content": "volume"})

        # 计算风险指数
        risk_series["risk_index"] = (
            (1 - risk_series["sentiment_score"].fillna(0.5)) * 50 +
            np.log1p(risk_series["volume"]) * 5
        )

        return risk_series.reset_index()


if __name__ == "__main__":
    # 测试
    analyzer = RiskPerceptionAnalyzer()

    # 创建测试数据
    test_data = pd.DataFrame({
        "content": [
            "疫情形势严峻，大家一定要做好防护",
            "经济下行压力大，很多企业面临困难",
            "今天天气不错，出去逛了逛街",
            "对这个政策很担心，不知道会有什么影响",
            "太生气了，这种事情怎么能发生"
        ],
        "created_at": pd.date_range("2024-01-01", periods=5, freq="H"),
        "reposts_count": [100, 50, 10, 200, 80]
    })

    result = analyzer.analyze_risk(test_data)

    print(f"综合风险得分: {result.overall_score:.1f}")
    print(f"风险等级: {result.risk_level.value}")
    print(f"维度得分: {result.dimension_scores}")
    print(f"关键因素: {result.key_factors}")
    print(f"趋势: {result.trend}")
    print(f"预警: {result.warnings}")
    print(f"建议: {result.recommendations}")

