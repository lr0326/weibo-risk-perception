"""
多维度风险感知模型
整合多种特征进行风险评估和预测
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config
from src.preprocessing.feature_extractor import FeatureExtractor
from src.analysis.sentiment_analyzer import SentimentAnalyzer


@dataclass
class RiskModelResult:
    """风险模型结果"""
    risk_level: str
    risk_probability: Dict[str, float]
    feature_importance: Dict[str, float]
    confidence: float


class MultiDimensionalRiskModel:
    """
    多维度风险感知模型

    功能：
    - 多维度特征融合
    - 风险等级分类
    - 特征重要性分析
    - 群体风险画像
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化风险模型

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 模型配置
        self.model_config = self.config.get("models", {}).get("risk_perception", {})
        self.dimensions = self.model_config.get("dimensions", [
            "health_risk", "economic_risk", "social_risk", "political_risk"
        ])

        # 特征提取器
        self.feature_extractor = FeatureExtractor(config_path)

        # 情感分析器
        self.sentiment_analyzer = SentimentAnalyzer(model_type="snownlp", config_path=config_path)

        # 分类器
        self.classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # 模型路径
        self.model_path = self.config.get("paths", {}).get("data", {}).get("models", "data/models")

        logger.info("多维度风险模型初始化完成")

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        include_sentiment: bool = True,
        include_tfidf: bool = False
    ) -> np.ndarray:
        """
        构建特征矩阵

        Args:
            df: 输入数据
            include_sentiment: 是否包含情感特征
            include_tfidf: 是否包含TF-IDF特征

        Returns:
            特征矩阵
        """
        # 情感分析
        if include_sentiment and "sentiment_score" not in df.columns:
            df = self.sentiment_analyzer.analyze_dataframe(df)

        # 提取特征
        features = self.feature_extractor.build_feature_matrix(
            df, include_tfidf=include_tfidf, normalize=False
        )

        return features

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost"
    ) -> Dict:
        """
        训练风险分类模型

        Args:
            X: 特征矩阵
            y: 标签
            model_type: 模型类型 (xgboost, random_forest, gradient_boosting)

        Returns:
            训练结果
        """
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 选择模型
        if model_type == "xgboost":
            self.classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )

        # 训练
        self.classifier.fit(X_train_scaled, y_train)

        # 评估
        y_pred = self.classifier.predict(X_test_scaled)

        # 交叉验证
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5)

        result = {
            "accuracy": float((y_pred == y_test).mean()),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }

        logger.info(f"模型训练完成 - 准确率: {result['accuracy']:.4f}")

        return result

    def predict(self, X: np.ndarray) -> List[RiskModelResult]:
        """
        预测风险等级

        Args:
            X: 特征矩阵

        Returns:
            预测结果列表
        """
        if self.classifier is None:
            raise ValueError("模型未训练，请先调用train方法")

        # 标准化
        X_scaled = self.scaler.transform(X)

        # 预测
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)

        # 特征重要性
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
        else:
            importances = np.zeros(X.shape[1])

        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            risk_level = self.label_encoder.inverse_transform([pred])[0]

            prob_dict = {
                self.label_encoder.inverse_transform([j])[0]: float(p)
                for j, p in enumerate(probs)
            }

            importance_dict = {
                f"feature_{j}": float(imp)
                for j, imp in enumerate(importances[:10])  # 前10个特征
            }

            results.append(RiskModelResult(
                risk_level=risk_level,
                risk_probability=prob_dict,
                feature_importance=importance_dict,
                confidence=float(max(probs))
            ))

        return results

    def segment_population(
        self,
        features: np.ndarray,
        n_clusters: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        群体风险感知细分

        Args:
            features: 特征矩阵
            n_clusters: 聚类数量

        Returns:
            聚类标签和群体画像
        """
        from sklearn.cluster import KMeans

        # 标准化
        features_scaled = StandardScaler().fit_transform(features)

        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        # 生成群体画像
        profiles = {}
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_features = features[mask]

            profiles[f"群体{cluster_id + 1}"] = {
                "size": int(mask.sum()),
                "percentage": float(mask.mean() * 100),
                "avg_features": cluster_features.mean(axis=0).tolist()[:5],
                "std_features": cluster_features.std(axis=0).tolist()[:5]
            }

        logger.info(f"群体细分完成，识别 {n_clusters} 个群体")

        return labels, profiles

    def analyze_risk_factors(
        self,
        df: pd.DataFrame,
        target_column: str = "risk_level"
    ) -> Dict:
        """
        分析风险因素

        Args:
            df: 输入数据
            target_column: 目标列

        Returns:
            风险因素分析结果
        """
        result = {
            "dimension_analysis": {},
            "correlation_analysis": {},
            "top_risk_factors": []
        }

        # 维度分析
        for dim in self.dimensions:
            if dim in df.columns:
                result["dimension_analysis"][dim] = {
                    "mean": float(df[dim].mean()),
                    "std": float(df[dim].std()),
                    "max": float(df[dim].max()),
                    "min": float(df[dim].min())
                }

        # 相关性分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            # 提取与目标变量的相关性
            if target_column in corr_matrix.columns:
                correlations = corr_matrix[target_column].drop(target_column, errors='ignore')
                result["correlation_analysis"] = correlations.to_dict()

                # 排序获取top因素
                sorted_corr = correlations.abs().sort_values(ascending=False)
                result["top_risk_factors"] = [
                    {"factor": k, "correlation": float(v)}
                    for k, v in sorted_corr.head(10).items()
                ]

        return result

    def save_model(self, filename: str = "risk_model.pkl"):
        """保存模型"""
        if self.classifier is None:
            logger.warning("模型未训练，无法保存")
            return

        filepath = os.path.join(self.model_path, filename)
        os.makedirs(self.model_path, exist_ok=True)

        model_data = {
            "classifier": self.classifier,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存至 {filepath}")

    def load_model(self, filename: str = "risk_model.pkl"):
        """加载模型"""
        filepath = os.path.join(self.model_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data["classifier"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]

        logger.info(f"模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试
    model = MultiDimensionalRiskModel()

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 20)
    y = np.random.choice(["low", "medium", "high", "critical"], n_samples)

    # 训练
    result = model.train(X, y, model_type="random_forest")
    print(f"训练结果: 准确率={result['accuracy']:.4f}")

    # 预测
    predictions = model.predict(X[:5])
    for i, pred in enumerate(predictions):
        print(f"样本{i+1}: {pred.risk_level} (置信度: {pred.confidence:.4f})")

    # 群体细分
    labels, profiles = model.segment_population(X, n_clusters=3)
    print(f"\n群体画像:")
    for name, profile in profiles.items():
        print(f"  {name}: {profile['size']}人 ({profile['percentage']:.1f}%)")

