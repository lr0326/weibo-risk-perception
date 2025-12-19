"""
社会网络分析模块
分析信息传播网络和用户关系网络
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import networkx as nx
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


@dataclass
class NetworkMetrics:
    """网络指标"""
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    avg_clustering: float
    diameter: Optional[int]
    avg_path_length: Optional[float]


@dataclass
class InfluencerInfo:
    """影响力用户信息"""
    user_id: str
    user_name: str
    followers: int
    degree_centrality: float
    betweenness_centrality: float
    pagerank: float
    influence_score: float


class NetworkAnalyzer:
    """
    社会网络分析器

    功能：
    - 构建传播网络
    - 计算网络指标
    - 识别关键传播者
    - 社区发现
    - 传播路径分析
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化网络分析器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.graph: Optional[nx.DiGraph] = None

        logger.info("网络分析器初始化完成")

    def build_repost_network(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        构建转发网络

        Args:
            df: 包含转发关系的数据

        Returns:
            有向图
        """
        G = nx.DiGraph()

        for _, row in df.iterrows():
            user_id = str(row.get("user_id", ""))

            if not user_id:
                continue

            # 添加节点
            G.add_node(user_id, **{
                "name": row.get("user_name", ""),
                "followers": row.get("user_followers", 0),
                "verified": row.get("user_verified", False)
            })

            # 如果是转发，添加边
            if row.get("is_repost", False):
                # 假设有原始用户ID字段
                original_user = str(row.get("original_user_id", ""))
                if original_user and original_user != user_id:
                    G.add_edge(original_user, user_id)

        self.graph = G
        logger.info(f"构建转发网络完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

        return G

    def build_interaction_network(self, df: pd.DataFrame) -> nx.Graph:
        """
        构建互动网络（基于@提及）

        Args:
            df: 包含微博内容的数据

        Returns:
            无向图
        """
        import re

        G = nx.Graph()
        mention_pattern = re.compile(r'@([\w\u4e00-\u9fff]+)')

        for _, row in df.iterrows():
            user_id = str(row.get("user_id", ""))
            content = row.get("content", "")

            if not user_id:
                continue

            # 添加用户节点
            G.add_node(user_id, **{
                "name": row.get("user_name", ""),
                "followers": row.get("user_followers", 0)
            })

            # 提取@提及
            mentions = mention_pattern.findall(content)
            for mentioned in mentions:
                if mentioned:
                    G.add_edge(user_id, mentioned)

        self.graph = G
        logger.info(f"构建互动网络完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

        return G

    def calculate_network_metrics(self, G: nx.Graph = None) -> NetworkMetrics:
        """
        计算网络指标

        Args:
            G: 网络图，默认使用self.graph

        Returns:
            NetworkMetrics对象
        """
        if G is None:
            G = self.graph

        if G is None or G.number_of_nodes() == 0:
            return NetworkMetrics(0, 0, 0, 0, 0, None, None)

        # 基础指标
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        density = nx.density(G)

        # 平均度
        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0

        # 聚类系数
        if isinstance(G, nx.DiGraph):
            G_undirected = G.to_undirected()
            avg_clustering = nx.average_clustering(G_undirected)
        else:
            avg_clustering = nx.average_clustering(G)

        # 直径和平均路径长度（仅对连通图）
        diameter = None
        avg_path_length = None

        try:
            if isinstance(G, nx.DiGraph):
                if nx.is_strongly_connected(G):
                    diameter = nx.diameter(G)
                    avg_path_length = nx.average_shortest_path_length(G)
            else:
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                    avg_path_length = nx.average_shortest_path_length(G)
        except:
            pass

        return NetworkMetrics(
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            avg_degree=avg_degree,
            avg_clustering=avg_clustering,
            diameter=diameter,
            avg_path_length=avg_path_length
        )

    def identify_influencers(
        self,
        G: nx.Graph = None,
        top_k: int = 10
    ) -> List[InfluencerInfo]:
        """
        识别关键影响者

        Args:
            G: 网络图
            top_k: 返回前k个

        Returns:
            影响者信息列表
        """
        if G is None:
            G = self.graph

        if G is None or G.number_of_nodes() == 0:
            return []

        # 计算中心性指标
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)

        # 综合评分
        influencers = []
        for node in G.nodes():
            node_data = G.nodes[node]

            # 综合影响力得分
            influence_score = (
                degree_centrality.get(node, 0) * 0.3 +
                betweenness_centrality.get(node, 0) * 0.3 +
                pagerank.get(node, 0) * 0.4
            )

            influencers.append(InfluencerInfo(
                user_id=str(node),
                user_name=node_data.get("name", ""),
                followers=node_data.get("followers", 0),
                degree_centrality=degree_centrality.get(node, 0),
                betweenness_centrality=betweenness_centrality.get(node, 0),
                pagerank=pagerank.get(node, 0),
                influence_score=influence_score
            ))

        # 按影响力排序
        influencers.sort(key=lambda x: x.influence_score, reverse=True)

        return influencers[:top_k]

    def detect_communities(
        self,
        G: nx.Graph = None,
        method: str = "louvain"
    ) -> Dict[str, int]:
        """
        社区发现

        Args:
            G: 网络图
            method: 算法 (louvain, label_propagation)

        Returns:
            节点到社区ID的映射
        """
        if G is None:
            G = self.graph

        if G is None or G.number_of_nodes() == 0:
            return {}

        # 转换为无向图
        if isinstance(G, nx.DiGraph):
            G = G.to_undirected()

        try:
            if method == "louvain":
                from networkx.algorithms.community import louvain_communities
                communities = louvain_communities(G)
            else:
                from networkx.algorithms.community import label_propagation_communities
                communities = label_propagation_communities(G)

            # 转换为节点-社区映射
            node_community = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_community[node] = i

            logger.info(f"发现 {len(set(node_community.values()))} 个社区")
            return node_community

        except Exception as e:
            logger.error(f"社区发现失败: {e}")
            return {}

    def analyze_propagation_path(
        self,
        G: nx.DiGraph = None,
        source: str = None
    ) -> Dict:
        """
        分析传播路径

        Args:
            G: 有向图
            source: 源节点

        Returns:
            传播路径分析结果
        """
        if G is None:
            G = self.graph

        if G is None or not isinstance(G, nx.DiGraph):
            return {}

        result = {
            "cascade_size": {},
            "cascade_depth": {},
            "avg_cascade_size": 0,
            "max_cascade_depth": 0
        }

        # 找到所有根节点（入度为0）
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]

        if source:
            roots = [source] if source in G else []

        for root in roots:
            # BFS计算级联大小和深度
            descendants = nx.descendants(G, root)
            cascade_size = len(descendants) + 1

            # 计算深度
            if descendants:
                depths = nx.single_source_shortest_path_length(G, root)
                cascade_depth = max(depths.values()) if depths else 0
            else:
                cascade_depth = 0

            result["cascade_size"][root] = cascade_size
            result["cascade_depth"][root] = cascade_depth

        if result["cascade_size"]:
            result["avg_cascade_size"] = np.mean(list(result["cascade_size"].values()))
            result["max_cascade_depth"] = max(result["cascade_depth"].values())

        return result

    def get_network_summary(self, G: nx.Graph = None) -> Dict:
        """
        获取网络摘要

        Args:
            G: 网络图

        Returns:
            摘要字典
        """
        if G is None:
            G = self.graph

        metrics = self.calculate_network_metrics(G)
        influencers = self.identify_influencers(G, top_k=5)

        return {
            "metrics": {
                "nodes": metrics.node_count,
                "edges": metrics.edge_count,
                "density": round(metrics.density, 4),
                "avg_degree": round(metrics.avg_degree, 2),
                "clustering": round(metrics.avg_clustering, 4)
            },
            "top_influencers": [
                {
                    "user_id": inf.user_id,
                    "user_name": inf.user_name,
                    "influence_score": round(inf.influence_score, 4)
                }
                for inf in influencers
            ]
        }

    def export_graph(self, G: nx.Graph = None, filepath: str = None) -> Dict:
        """
        导出网络数据

        Args:
            G: 网络图
            filepath: 保存路径（可选）

        Returns:
            JSON格式的网络数据
        """
        if G is None:
            G = self.graph

        if G is None:
            return {"nodes": [], "edges": []}

        data = {
            "nodes": [
                {"id": str(n), **G.nodes[n]}
                for n in G.nodes()
            ],
            "edges": [
                {"source": str(u), "target": str(v)}
                for u, v in G.edges()
            ]
        }

        if filepath:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"网络数据已导出至 {filepath}")

        return data


if __name__ == "__main__":
    # 测试
    analyzer = NetworkAnalyzer()

    # 创建测试网络
    G = nx.DiGraph()
    G.add_edge("user1", "user2")
    G.add_edge("user1", "user3")
    G.add_edge("user2", "user4")
    G.add_edge("user2", "user5")
    G.add_edge("user3", "user6")

    for node in G.nodes():
        G.nodes[node]["name"] = f"用户{node[-1]}"
        G.nodes[node]["followers"] = np.random.randint(100, 10000)

    analyzer.graph = G

    # 计算指标
    metrics = analyzer.calculate_network_metrics()
    print(f"网络指标: {metrics}")

    # 识别影响者
    influencers = analyzer.identify_influencers(top_k=3)
    print(f"\n关键影响者:")
    for inf in influencers:
        print(f"  {inf.user_name}: 影响力={inf.influence_score:.4f}")

    # 网络摘要
    summary = analyzer.get_network_summary()
    print(f"\n网络摘要: {summary}")

