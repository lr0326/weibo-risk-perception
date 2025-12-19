"""
特征提取模块
提供文本特征、用户特征、传播特征等多维度特征提取
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config
from src.preprocessing.text_cleaner import TextCleaner


class FeatureExtractor:
    """
    特征提取器

    支持特征类型：
    - 文本特征：TF-IDF、词向量、n-gram
    - 情感特征：情感词统计、情感强度
    - 用户特征：影响力、活跃度
    - 传播特征：转发深度、传播速度
    - 时间特征：发布时间、周期性
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化特征提取器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 特征提取配置
        feature_config = self.config.get("preprocessing", {}).get("feature_extraction", {})

        # TF-IDF配置
        tfidf_config = feature_config.get("tfidf", {})
        self.tfidf_max_features = tfidf_config.get("max_features", 5000)
        self.tfidf_ngram_range = tuple(tfidf_config.get("ngram_range", [1, 2]))

        # 文本清洗器
        self.text_cleaner = TextCleaner(config_path)

        # TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            tokenizer=self.text_cleaner.tokenize,
            token_pattern=None
        )

        # 计数向量化器
        self.count_vectorizer = CountVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            tokenizer=self.text_cleaner.tokenize,
            token_pattern=None
        )

        # 标准化器
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()

        # 情感词典
        self.positive_words: set = set()
        self.negative_words: set = set()
        self._load_sentiment_dict()

        logger.info("特征提取器初始化完成")

    def _load_sentiment_dict(self):
        """加载情感词典"""
        # 默认情感词（简化版）
        self.positive_words = {
            '好', '棒', '赞', '喜欢', '开心', '高兴', '满意', '优秀', '出色',
            '感谢', '支持', '希望', '期待', '成功', '进步', '发展', '改善',
            '美丽', '温暖', '友好', '积极', '乐观', '正能量'
        }

        self.negative_words = {
            '差', '糟', '烂', '讨厌', '难过', '失望', '愤怒', '生气', '担心',
            '害怕', '恐惧', '焦虑', '悲伤', '痛苦', '困难', '问题', '危机',
            '风险', '威胁', '损失', '负面', '消极', '失败'
        }

        logger.debug(f"加载情感词: 正面{len(self.positive_words)}个, 负面{len(self.negative_words)}个")

    def extract_tfidf_features(
        self,
        texts: List[str],
        fit: bool = True
    ) -> np.ndarray:
        """
        提取TF-IDF特征

        Args:
            texts: 文本列表
            fit: 是否拟合向量化器

        Returns:
            TF-IDF特征矩阵
        """
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)

        logger.debug(f"TF-IDF特征维度: {tfidf_matrix.shape}")
        return tfidf_matrix.toarray()

    def extract_text_statistics(self, text: str) -> Dict[str, float]:
        """
        提取文本统计特征

        Args:
            text: 输入文本

        Returns:
            统计特征字典
        """
        cleaned = self.text_cleaner.clean(text)
        tokens = self.text_cleaner.tokenize(text)

        # 基础统计
        char_count = len(cleaned)
        word_count = len(tokens)
        unique_words = len(set(tokens))

        # 词汇多样性
        lexical_diversity = unique_words / max(word_count, 1)

        # 平均词长
        avg_word_length = np.mean([len(w) for w in tokens]) if tokens else 0

        # 标点符号密度
        punctuation_count = sum(1 for c in text if c in '，。！？、；：""''')
        punctuation_density = punctuation_count / max(char_count, 1)

        return {
            "char_count": char_count,
            "word_count": word_count,
            "unique_words": unique_words,
            "lexical_diversity": lexical_diversity,
            "avg_word_length": avg_word_length,
            "punctuation_density": punctuation_density
        }

    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        提取情感特征

        Args:
            text: 输入文本

        Returns:
            情感特征字典
        """
        tokens = set(self.text_cleaner.tokenize(text))

        # 情感词统计
        positive_count = len(tokens & self.positive_words)
        negative_count = len(tokens & self.negative_words)
        total_sentiment = positive_count + negative_count

        # 情感比例
        if total_sentiment > 0:
            positive_ratio = positive_count / total_sentiment
            negative_ratio = negative_count / total_sentiment
        else:
            positive_ratio = 0.5
            negative_ratio = 0.5

        # 情感极性得分 (-1 到 1)
        polarity = (positive_count - negative_count) / max(len(tokens), 1)

        # 情感强度
        intensity = total_sentiment / max(len(tokens), 1)

        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "polarity": polarity,
            "intensity": intensity
        }

    def extract_user_features(self, row: pd.Series) -> Dict[str, float]:
        """
        提取用户特征

        Args:
            row: 包含用户信息的数据行

        Returns:
            用户特征字典
        """
        followers = row.get("user_followers", 0)
        verified = row.get("user_verified", False)

        # 影响力得分（对数变换）
        influence_score = np.log1p(followers)

        # 认证加权
        verified_weight = 1.5 if verified else 1.0

        # 综合影响力
        overall_influence = influence_score * verified_weight

        return {
            "followers_log": influence_score,
            "verified": float(verified),
            "influence_score": overall_influence
        }

    def extract_engagement_features(self, row: pd.Series) -> Dict[str, float]:
        """
        提取互动特征

        Args:
            row: 包含互动数据的数据行

        Returns:
            互动特征字典
        """
        reposts = row.get("reposts_count", 0)
        comments = row.get("comments_count", 0)
        attitudes = row.get("attitudes_count", 0)

        # 总互动量
        total_engagement = reposts + comments + attitudes

        # 对数变换
        reposts_log = np.log1p(reposts)
        comments_log = np.log1p(comments)
        attitudes_log = np.log1p(attitudes)
        engagement_log = np.log1p(total_engagement)

        # 互动类型比例
        if total_engagement > 0:
            repost_ratio = reposts / total_engagement
            comment_ratio = comments / total_engagement
            attitude_ratio = attitudes / total_engagement
        else:
            repost_ratio = comment_ratio = attitude_ratio = 0

        # 传播力指标（转发权重更高）
        virality_score = reposts_log * 2 + comments_log * 1.5 + attitudes_log

        return {
            "total_engagement": total_engagement,
            "engagement_log": engagement_log,
            "reposts_log": reposts_log,
            "comments_log": comments_log,
            "attitudes_log": attitudes_log,
            "repost_ratio": repost_ratio,
            "comment_ratio": comment_ratio,
            "attitude_ratio": attitude_ratio,
            "virality_score": virality_score
        }

    def extract_temporal_features(self, timestamp: datetime) -> Dict[str, float]:
        """
        提取时间特征

        Args:
            timestamp: 时间戳

        Returns:
            时间特征字典
        """
        if timestamp is None:
            return {
                "hour": 0, "day_of_week": 0, "is_weekend": 0,
                "is_business_hour": 0, "hour_sin": 0, "hour_cos": 0
            }

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = float(day_of_week >= 5)
        is_business_hour = float(9 <= hour <= 18)

        # 周期性编码
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        return {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_business_hour": is_business_hour,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos
        }

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有特征

        Args:
            df: 输入DataFrame

        Returns:
            包含所有特征的DataFrame
        """
        features_list = []

        for idx, row in df.iterrows():
            features = {}

            # 文本特征
            text = row.get("content", "")
            features.update(self.extract_text_statistics(text))
            features.update(self.extract_sentiment_features(text))

            # 用户特征
            features.update(self.extract_user_features(row))

            # 互动特征
            features.update(self.extract_engagement_features(row))

            # 时间特征
            timestamp = row.get("created_at")
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    timestamp = None
            features.update(self.extract_temporal_features(timestamp))

            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        logger.info(f"提取特征完成，维度: {features_df.shape}")

        return features_df

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        include_tfidf: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        构建特征矩阵

        Args:
            df: 输入DataFrame
            include_tfidf: 是否包含TF-IDF特征
            normalize: 是否标准化

        Returns:
            特征矩阵
        """
        # 提取数值特征
        features_df = self.extract_all_features(df)
        numerical_features = features_df.values

        if include_tfidf:
            # TF-IDF特征
            texts = df["content"].fillna("").tolist()
            tfidf_features = self.extract_tfidf_features(texts)

            # 合并特征
            feature_matrix = np.hstack([numerical_features, tfidf_features])
        else:
            feature_matrix = numerical_features

        if normalize:
            feature_matrix = self.scaler.fit_transform(feature_matrix)

        logger.info(f"特征矩阵维度: {feature_matrix.shape}")
        return feature_matrix

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        sample_features = {
            **self.extract_text_statistics("示例文本"),
            **self.extract_sentiment_features("示例文本"),
            **self.extract_user_features(pd.Series({})),
            **self.extract_engagement_features(pd.Series({})),
            **self.extract_temporal_features(datetime.now())
        }

        feature_names = list(sample_features.keys())

        # 添加TF-IDF特征名
        if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            try:
                tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
                feature_names.extend([f"tfidf_{name}" for name in tfidf_names])
            except:
                pass

        return feature_names


if __name__ == "__main__":
    # 测试
    extractor = FeatureExtractor()

    # 测试单条文本
    test_text = "今天天气真好，希望疫情早日结束，大家都能恢复正常生活。"

    print("文本统计特征:")
    print(extractor.extract_text_statistics(test_text))

    print("\n情感特征:")
    print(extractor.extract_sentiment_features(test_text))

    print("\n时间特征:")
    print(extractor.extract_temporal_features(datetime.now()))

