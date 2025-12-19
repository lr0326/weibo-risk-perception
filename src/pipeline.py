"""
主流程模块
整合所有功能的完整分析流程
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config, Timer
from src.utils.logger import setup_logger
from src.data_collection.weibo_collector import WeiboDataCollector, MockWeiboDataGenerator
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.feature_extractor import FeatureExtractor
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.risk_perception import RiskPerceptionAnalyzer, RiskAssessment
from src.analysis.network_analysis import NetworkAnalyzer
from src.models.risk_model import MultiDimensionalRiskModel
from src.models.prediction_model import EmotionTrendPredictor
from src.models.clustering import PopulationClusteringModel

# 可视化模块可选
try:
    from src.visualization.report_generator import ReportGenerator
    HAS_REPORT_GENERATOR = True
except ImportError:
    ReportGenerator = None
    HAS_REPORT_GENERATOR = False


class RiskPerceptionPipeline:
    """
    风险感知分析流水线

    整合数据采集、预处理、分析、建模、可视化的完整流程
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化流水线

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.config_path = config_path

        # 初始化日志
        setup_logger()

        # 初始化组件
        self.text_cleaner = TextCleaner(config_path)
        self.feature_extractor = FeatureExtractor(config_path)
        self.sentiment_analyzer = SentimentAnalyzer(model_type="snownlp", config_path=config_path)
        self.risk_analyzer = RiskPerceptionAnalyzer(config_path)
        self.network_analyzer = NetworkAnalyzer(config_path)
        self.risk_model = MultiDimensionalRiskModel(config_path)
        self.predictor = EmotionTrendPredictor(method="lstm", config_path=config_path)
        self.clustering_model = PopulationClusteringModel(config_path)
        self.report_generator = ReportGenerator(config_path) if HAS_REPORT_GENERATOR else None

        # 数据存储
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.results: Dict = {}

        logger.info("风险感知分析流水线初始化完成")

    def collect_data(
        self,
        keywords: str = None,
        count: int = 100,
        pages: int = 10,
        use_mock: bool = True,
        access_token: str = None
    ) -> pd.DataFrame:
        """
        数据采集

        Args:
            keywords: 搜索关键词
            count: 每页数量
            pages: 采集页数
            use_mock: 是否使用模拟数据
            access_token: API访问令牌

        Returns:
            采集的数据DataFrame
        """
        with Timer("数据采集"):
            if use_mock:
                logger.info("使用模拟数据生成器")
                generator = MockWeiboDataGenerator()
                self.raw_data = generator.generate_mock_data(count * pages)
            else:
                collector = WeiboDataCollector(
                    access_token=access_token,
                    config_path=self.config_path
                )
                self.raw_data = collector.search_weibo_by_keyword(
                    keyword=keywords,
                    count=count,
                    pages=pages
                )

        logger.info(f"采集数据: {len(self.raw_data)} 条")
        return self.raw_data

    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        数据预处理

        Args:
            df: 输入数据，默认使用raw_data

        Returns:
            预处理后的数据
        """
        if df is None:
            df = self.raw_data

        if df is None or df.empty:
            logger.warning("没有数据可供预处理")
            return pd.DataFrame()

        with Timer("数据预处理"):
            # 文本清洗
            df = self.text_cleaner.process_dataframe(
                df,
                text_column="content",
                output_column="cleaned_content",
                tokenize=True,
                token_column="tokens"
            )

            self.processed_data = df

        logger.info(f"预处理完成: {len(df)} 条")
        return df

    def analyze_sentiment(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        情感分析

        Args:
            df: 输入数据

        Returns:
            添加情感分析结果的数据
        """
        if df is None:
            df = self.processed_data

        if df is None or df.empty:
            logger.warning("没有数据可供分析")
            return pd.DataFrame()

        with Timer("情感分析"):
            df = self.sentiment_analyzer.analyze_dataframe(df, text_column="cleaned_content")
            self.processed_data = df

        return df

    def assess_risk(self, df: pd.DataFrame = None) -> RiskAssessment:
        """
        风险评估

        Args:
            df: 输入数据

        Returns:
            风险评估结果
        """
        if df is None:
            df = self.processed_data

        if df is None or df.empty:
            logger.warning("没有数据可供评估")
            return self.risk_analyzer._empty_assessment()

        with Timer("风险评估"):
            assessment = self.risk_analyzer.analyze_risk(df)

        # 保存结果
        self.results["risk_assessment"] = assessment

        return assessment

    def predict_trend(
        self,
        df: pd.DataFrame = None,
        steps: int = 24,
        train: bool = True
    ) -> Dict:
        """
        趋势预测

        Args:
            df: 输入数据
            steps: 预测步数
            train: 是否训练模型

        Returns:
            预测结果
        """
        if df is None:
            df = self.processed_data

        if df is None or df.empty or len(df) < 50:
            logger.warning("数据量不足，跳过趋势预测")
            return {}

        with Timer("趋势预测"):
            # 准备时间序列
            try:
                ts = self.predictor.prepare_time_series(df, freq="H")

                if len(ts) < 30:
                    logger.warning("时间序列数据不足")
                    return {}

                # 训练模型
                if train:
                    train_result = self.predictor.train_lstm(ts, epochs=30)
                    logger.info(f"模型训练完成: RMSE={train_result['val_rmse']:.4f}")

                # 预测
                predictions = self.predictor.predict_future(ts, steps=steps)

                self.results["predictions"] = predictions

                return {
                    "timestamps": predictions.timestamps,
                    "predictions": predictions.predictions,
                    "confidence_lower": predictions.confidence_lower,
                    "confidence_upper": predictions.confidence_upper
                }
            except Exception as e:
                logger.error(f"趋势预测失败: {e}")
                return {}

    def cluster_population(
        self,
        df: pd.DataFrame = None,
        n_clusters: int = 5
    ) -> Dict:
        """
        群体聚类

        Args:
            df: 输入数据
            n_clusters: 聚类数量

        Returns:
            聚类结果
        """
        if df is None:
            df = self.processed_data

        if df is None or df.empty:
            logger.warning("没有数据可供聚类")
            return {}

        with Timer("群体聚类"):
            # 提取特征
            features = self.feature_extractor.extract_all_features(df)
            feature_matrix = features.values

            # 聚类
            labels = self.clustering_model.fit(feature_matrix, n_clusters=n_clusters)

            # 生成画像
            feature_names = list(features.columns)
            profiles = self.clustering_model.generate_cluster_profiles(
                feature_matrix, feature_names
            )

            self.results["clustering"] = {
                "labels": labels.tolist(),
                "profiles": profiles
            }

            return {
                "n_clusters": len(profiles),
                "profiles": [
                    {
                        "cluster_id": p.cluster_id,
                        "size": p.size,
                        "percentage": p.percentage,
                        "description": p.description
                    }
                    for p in profiles
                ]
            }

    def analyze_network(self, df: pd.DataFrame = None) -> Dict:
        """
        网络分析

        Args:
            df: 输入数据

        Returns:
            网络分析结果
        """
        if df is None:
            df = self.processed_data

        if df is None or df.empty:
            return {}

        with Timer("网络分析"):
            # 构建互动网络
            G = self.network_analyzer.build_interaction_network(df)

            # 获取摘要
            summary = self.network_analyzer.get_network_summary(G)

            self.results["network"] = summary

            return summary

    def run_full_analysis(
        self,
        keywords: str = None,
        count: int = 100,
        pages: int = 5,
        use_mock: bool = True,
        access_token: str = None
    ) -> Dict:
        """
        运行完整分析流程

        Args:
            keywords: 搜索关键词
            count: 每页数量
            pages: 采集页数
            use_mock: 是否使用模拟数据
            access_token: API访问令牌

        Returns:
            完整分析结果
        """
        logger.info("="*50)
        logger.info("开始完整分析流程")
        logger.info("="*50)

        # 1. 数据采集
        df = self.collect_data(
            keywords=keywords,
            count=count,
            pages=pages,
            use_mock=use_mock,
            access_token=access_token
        )

        if df.empty:
            logger.error("数据采集失败")
            return {}

        # 2. 数据预处理
        df = self.preprocess_data(df)

        # 3. 情感分析
        df = self.analyze_sentiment(df)

        # 4. 风险评估
        risk_assessment = self.assess_risk(df)

        # 5. 趋势预测
        predictions = self.predict_trend(df, steps=24, train=True)

        # 6. 群体聚类
        clustering = self.cluster_population(df, n_clusters=4)

        # 7. 网络分析
        network = self.analyze_network(df)

        # 汇总结果
        self.results["summary"] = {
            "analysis_date": datetime.now().isoformat(),
            "sample_size": len(df),
            "risk_level": risk_assessment.risk_level.value,
            "risk_score": risk_assessment.overall_score,
            "dimension_scores": risk_assessment.dimension_scores,
            "trend": risk_assessment.trend,
            "warnings": risk_assessment.warnings,
            "recommendations": risk_assessment.recommendations,
            "sentiment_summary": self._get_sentiment_summary(df),
            "clustering": clustering,
            "network": network,
            "predictions": predictions
        }

        logger.info("="*50)
        logger.info("完整分析流程完成")
        logger.info("="*50)

        return self.results["summary"]

    def _get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """获取情感分析摘要"""
        if "sentiment_score" not in df.columns:
            return {}

        return {
            "avg_polarity": float(df["sentiment_score"].mean()),
            "dominant_emotion": df.get("emotion", pd.Series(["neutral"])).mode().iloc[0] if "emotion" in df.columns else "neutral",
            "avg_intensity": float(df.get("sentiment_intensity", pd.Series([0.5])).mean()),
            "polarity_distribution": df["sentiment_polarity"].value_counts().to_dict() if "sentiment_polarity" in df.columns else {}
        }

    def generate_report(
        self,
        output_path: str = None,
        format: str = "html"
    ) -> str:
        """
        生成分析报告

        Args:
            output_path: 输出路径
            format: 输出格式

        Returns:
            报告文件路径
        """
        if "summary" not in self.results:
            logger.warning("没有分析结果，请先运行分析")
            return ""

        if self.report_generator is None:
            logger.warning("报告生成器不可用，请安装 jinja2")
            return ""

        return self.report_generator.generate_report(
            self.results["summary"],
            output_path=output_path,
            format=format
        )

    def save_data(self, output_dir: str = "data/outputs"):
        """
        保存处理后的数据

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.processed_data is not None:
            filepath = os.path.join(output_dir, f"processed_data_{timestamp}.csv")
            self.processed_data.to_csv(filepath, index=False, encoding="utf-8-sig")
            logger.info(f"数据已保存至 {filepath}")

        if self.results:
            import json
            filepath = os.path.join(output_dir, f"results_{timestamp}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.results.get("summary", {}), f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"结果已保存至 {filepath}")


def main():
    """主函数"""
    # 初始化流水线
    pipeline = RiskPerceptionPipeline()

    # 运行完整分析
    results = pipeline.run_full_analysis(
        keywords="社会热点",
        count=50,
        pages=2,
        use_mock=True
    )

    # 打印结果
    print("\n" + "="*50)
    print("分析结果摘要")
    print("="*50)
    print(f"样本量: {results.get('sample_size', 0)}")
    print(f"风险等级: {results.get('risk_level', '-')}")
    print(f"风险得分: {results.get('risk_score', 0):.1f}")
    print(f"趋势: {results.get('trend', '-')}")

    print("\n预警信息:")
    for warning in results.get('warnings', []):
        print(f"  {warning}")

    print("\n应对建议:")
    for rec in results.get('recommendations', []):
        print(f"  {rec}")

    # 生成报告
    report_path = pipeline.generate_report(format="html")
    print(f"\n报告已生成: {report_path}")

    # 保存数据
    pipeline.save_data()


if __name__ == "__main__":
    main()

