"""
聚类分析模块
用于群体细分和用户画像
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pickle
import os

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


@dataclass
class ClusterProfile:
    """聚类画像"""
    cluster_id: int
    size: int
    percentage: float
    centroid: List[float]
    characteristics: Dict[str, float]
    description: str


class PopulationClusteringModel:
    """
    群体聚类模型

    功能：
    - 多种聚类算法支持
    - 最优聚类数选择
    - 群体画像生成
    - 聚类可视化
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化聚类模型

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 聚类配置
        cluster_config = self.config.get("models", {}).get("clustering", {})
        self.algorithm = cluster_config.get("algorithm", "kmeans")
        self.n_clusters = cluster_config.get("n_clusters", 5)
        self.random_state = cluster_config.get("random_state", 42)

        # 模型
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None

        # 结果
        self.labels_ = None
        self.cluster_centers_ = None

        # 模型路径
        self.model_path = self.config.get("paths", {}).get("data", {}).get("models", "data/models")

        logger.info(f"群体聚类模型初始化完成，算法: {self.algorithm}")

    def fit(
        self,
        X: np.ndarray,
        algorithm: str = None,
        n_clusters: int = None,
        **kwargs
    ) -> np.ndarray:
        """
        执行聚类

        Args:
            X: 特征矩阵
            algorithm: 聚类算法
            n_clusters: 聚类数
            **kwargs: 其他参数

        Returns:
            聚类标签
        """
        if algorithm is None:
            algorithm = self.algorithm
        if n_clusters is None:
            n_clusters = self.n_clusters

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 选择算法
        if algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                **kwargs
            )
        elif algorithm == "dbscan":
            eps = kwargs.get("eps", 0.5)
            min_samples = kwargs.get("min_samples", 5)
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == "hierarchical":
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        # 聚类
        self.labels_ = self.model.fit_predict(X_scaled)

        # 保存聚类中心
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers_ = self.scaler.inverse_transform(
                self.model.cluster_centers_
            )

        n_clusters_found = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        logger.info(f"聚类完成，发现 {n_clusters_found} 个簇")

        return self.labels_

    def find_optimal_clusters(
        self,
        X: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 10
    ) -> Dict:
        """
        寻找最优聚类数

        Args:
            X: 特征矩阵
            min_clusters: 最小聚类数
            max_clusters: 最大聚类数

        Returns:
            评估结果
        """
        X_scaled = self.scaler.fit_transform(X)

        results = {
            "n_clusters": [],
            "silhouette": [],
            "calinski_harabasz": [],
            "inertia": []
        }

        for k in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            results["n_clusters"].append(k)
            results["silhouette"].append(silhouette_score(X_scaled, labels))
            results["calinski_harabasz"].append(calinski_harabasz_score(X_scaled, labels))
            results["inertia"].append(kmeans.inertia_)

        # 找到最优k（基于轮廓系数）
        optimal_k = results["n_clusters"][np.argmax(results["silhouette"])]

        logger.info(f"最优聚类数: {optimal_k}")

        return {
            "optimal_k": optimal_k,
            "metrics": results
        }

    def generate_cluster_profiles(
        self,
        X: np.ndarray,
        feature_names: List[str] = None,
        df: pd.DataFrame = None
    ) -> List[ClusterProfile]:
        """
        生成聚类画像

        Args:
            X: 特征矩阵
            feature_names: 特征名称
            df: 原始数据（可选）

        Returns:
            聚类画像列表
        """
        if self.labels_ is None:
            raise ValueError("请先执行聚类")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        profiles = []
        unique_labels = sorted(set(self.labels_))

        for cluster_id in unique_labels:
            if cluster_id == -1:  # DBSCAN噪声点
                continue

            mask = self.labels_ == cluster_id
            cluster_data = X[mask]

            # 基本统计
            size = int(mask.sum())
            percentage = float(size / len(X) * 100)
            centroid = cluster_data.mean(axis=0).tolist()

            # 特征分析
            characteristics = {}
            for i, name in enumerate(feature_names[:10]):  # 取前10个特征
                characteristics[name] = float(cluster_data[:, i].mean())

            # 生成描述
            description = self._generate_description(cluster_id, characteristics, size)

            profiles.append(ClusterProfile(
                cluster_id=cluster_id,
                size=size,
                percentage=percentage,
                centroid=centroid,
                characteristics=characteristics,
                description=description
            ))

        return profiles

    def _generate_description(
        self,
        cluster_id: int,
        characteristics: Dict[str, float],
        size: int
    ) -> str:
        """生成聚类描述"""
        # 简单的规则化描述
        descriptions = []

        # 基于特征值生成描述
        if "polarity" in characteristics:
            if characteristics["polarity"] > 0.3:
                descriptions.append("情感偏正面")
            elif characteristics["polarity"] < -0.3:
                descriptions.append("情感偏负面")
            else:
                descriptions.append("情感中性")

        if "intensity" in characteristics:
            if characteristics["intensity"] > 0.6:
                descriptions.append("情感强烈")
            elif characteristics["intensity"] < 0.3:
                descriptions.append("情感平淡")

        if "engagement_log" in characteristics:
            if characteristics["engagement_log"] > 5:
                descriptions.append("高互动群体")
            elif characteristics["engagement_log"] < 2:
                descriptions.append("低互动群体")

        if "influence_score" in characteristics:
            if characteristics["influence_score"] > 5:
                descriptions.append("高影响力")

        if not descriptions:
            descriptions.append(f"群体{cluster_id + 1}")

        return f"({size}人) " + "、".join(descriptions)

    def reduce_dimensions(
        self,
        X: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        降维用于可视化

        Args:
            X: 特征矩阵
            n_components: 目标维度

        Returns:
            降维后的数据
        """
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X_scaled)

        explained_variance = self.pca.explained_variance_ratio_.sum()
        logger.info(f"降维完成，解释方差比: {explained_variance:.4f}")

        return X_reduced

    def get_cluster_summary(self) -> Dict:
        """获取聚类摘要"""
        if self.labels_ is None:
            return {}

        unique_labels = sorted(set(self.labels_))
        summary = {
            "n_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "cluster_sizes": {},
            "noise_points": 0
        }

        for label in unique_labels:
            count = int((self.labels_ == label).sum())
            if label == -1:
                summary["noise_points"] = count
            else:
                summary["cluster_sizes"][f"cluster_{label}"] = count

        return summary

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据的聚类标签

        Args:
            X: 特征矩阵

        Returns:
            聚类标签
        """
        if self.model is None:
            raise ValueError("模型未训练")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, 'predict'):
            return self.model.predict(X_scaled)
        else:
            # 对于没有predict方法的算法，使用最近邻
            from sklearn.neighbors import NearestNeighbors

            if self.cluster_centers_ is not None:
                centers_scaled = self.scaler.transform(self.cluster_centers_)
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(centers_scaled)
                _, indices = nn.kneighbors(X_scaled)
                return indices.flatten()

            return np.zeros(len(X))

    def save_model(self, filename: str = "clustering_model.pkl"):
        """保存模型"""
        if self.model is None:
            logger.warning("模型未训练，无法保存")
            return

        filepath = os.path.join(self.model_path, filename)
        os.makedirs(self.model_path, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "labels": self.labels_,
            "cluster_centers": self.cluster_centers_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存至 {filepath}")

    def load_model(self, filename: str = "clustering_model.pkl"):
        """加载模型"""
        filepath = os.path.join(self.model_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.labels_ = model_data["labels"]
        self.cluster_centers_ = model_data["cluster_centers"]

        logger.info(f"模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试
    model = PopulationClusteringModel()

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200

    # 生成3个簇
    cluster1 = np.random.randn(70, 10) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(70, 10) + np.array([-2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(60, 10) + np.array([0, -2, 2, 0, 0, 0, 0, 0, 0, 0])

    X = np.vstack([cluster1, cluster2, cluster3])

    # 寻找最优k
    optimal_result = model.find_optimal_clusters(X, min_clusters=2, max_clusters=6)
    print(f"最优聚类数: {optimal_result['optimal_k']}")

    # 聚类
    labels = model.fit(X, n_clusters=3)
    print(f"\n聚类摘要: {model.get_cluster_summary()}")

    # 生成画像
    feature_names = ["polarity", "intensity", "engagement_log"] + [f"feat_{i}" for i in range(7)]
    profiles = model.generate_cluster_profiles(X, feature_names)

    print("\n群体画像:")
    for profile in profiles:
        print(f"  簇{profile.cluster_id}: {profile.description}")

