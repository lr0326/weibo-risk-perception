"""
情感分析模块
提供多种情感分析方法：BERT、LSTM、TextCNN、SnowNLP等
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


class SentimentLabel(Enum):
    """情感标签枚举"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class EmotionType(Enum):
    """情绪类型枚举"""
    JOY = "joy"
    ANGER = "anger"
    FEAR = "fear"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """情感分析结果"""
    text: str
    polarity: str  # positive, neutral, negative
    polarity_score: float  # -1 到 1
    emotion: str
    emotion_scores: Dict[str, float]
    intensity: float  # 0 到 1
    confidence: float


class SentimentAnalyzer:
    """
    情感分析器

    支持多种分析方法：
    - BERT: 基于预训练BERT模型的情感分析
    - LSTM: 基于LSTM的情感分类
    - SnowNLP: 基于SnowNLP的中文情感分析
    - Dictionary: 基于情感词典的分析
    """

    def __init__(
        self,
        model_type: str = "snownlp",
        config_path: str = "config/config.yaml"
    ):
        """
        初始化情感分析器

        Args:
            model_type: 模型类型 (bert, lstm, snownlp, dictionary)
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.model_type = model_type.lower()

        # 模型配置
        model_config = self.config.get("models", {}).get("sentiment", {})
        self.model_name = model_config.get("model_name", "bert-base-chinese")
        self.max_length = model_config.get("max_length", 128)
        self.batch_size = model_config.get("batch_size", 32)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self._init_model()

        # 情感词典
        self._load_sentiment_lexicon()

        logger.info(f"情感分析器初始化完成，使用模型: {self.model_type}")

    def _init_model(self):
        """初始化模型"""
        if self.model_type == "bert":
            try:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3
                ).to(self.device)
                self.model.eval()
                logger.info("BERT模型加载成功")
            except Exception as e:
                logger.warning(f"BERT模型加载失败，降级使用SnowNLP: {e}")
                self.model_type = "snownlp"

        elif self.model_type == "snownlp":
            try:
                from snownlp import SnowNLP
                self.snownlp_class = SnowNLP
                logger.info("SnowNLP初始化成功")
            except ImportError:
                logger.warning("SnowNLP未安装，降级使用词典方法")
                self.model_type = "dictionary"

    def _load_sentiment_lexicon(self):
        """加载情感词典"""
        # 正面情感词
        self.positive_words = {
            '好': 1.0, '棒': 1.0, '赞': 1.0, '喜欢': 0.8, '开心': 0.9,
            '高兴': 0.9, '满意': 0.8, '优秀': 0.9, '出色': 0.9, '感谢': 0.7,
            '支持': 0.7, '希望': 0.6, '期待': 0.6, '成功': 0.8, '进步': 0.7,
            '发展': 0.6, '改善': 0.7, '美丽': 0.7, '温暖': 0.7, '友好': 0.6,
            '积极': 0.7, '乐观': 0.8, '正能量': 0.9, '幸福': 0.9, '快乐': 0.9
        }

        # 负面情感词
        self.negative_words = {
            '差': -1.0, '糟': -1.0, '烂': -1.0, '讨厌': -0.8, '难过': -0.7,
            '失望': -0.8, '愤怒': -0.9, '生气': -0.8, '担心': -0.6, '害怕': -0.8,
            '恐惧': -0.9, '焦虑': -0.7, '悲伤': -0.8, '痛苦': -0.9, '困难': -0.5,
            '问题': -0.4, '危机': -0.8, '风险': -0.6, '威胁': -0.7, '损失': -0.7,
            '负面': -0.6, '消极': -0.6, '失败': -0.8, '绝望': -1.0, '崩溃': -0.9
        }

        # 程度副词
        self.degree_words = {
            '非常': 1.5, '特别': 1.5, '极其': 1.8, '十分': 1.4, '太': 1.3,
            '很': 1.2, '挺': 1.1, '比较': 0.8, '有点': 0.6, '稍微': 0.5,
            '略微': 0.4
        }

        # 否定词
        self.negation_words = {'不', '没', '无', '非', '未', '别', '勿', '莫', '否'}

        # 情绪词典
        self.emotion_words = {
            'joy': {'开心', '高兴', '快乐', '幸福', '喜悦', '欢乐', '愉快'},
            'anger': {'愤怒', '生气', '气愤', '恼火', '暴怒', '发火', '怒火'},
            'fear': {'害怕', '恐惧', '担心', '忧虑', '恐慌', '惊恐', '畏惧'},
            'sadness': {'难过', '悲伤', '伤心', '痛苦', '忧伤', '哀伤', '悲痛'},
            'surprise': {'惊讶', '震惊', '意外', '吃惊', '诧异', '惊奇'},
            'disgust': {'厌恶', '恶心', '讨厌', '反感', '嫌弃', '作呕'}
        }

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        分析单条文本的情感

        Args:
            text: 输入文本

        Returns:
            SentimentResult对象
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                polarity="neutral",
                polarity_score=0.0,
                emotion="neutral",
                emotion_scores={e.value: 0.0 for e in EmotionType},
                intensity=0.0,
                confidence=0.0
            )

        if self.model_type == "bert":
            return self._analyze_with_bert(text)
        elif self.model_type == "snownlp":
            return self._analyze_with_snownlp(text)
        else:
            return self._analyze_with_dictionary(text)

    def _analyze_with_bert(self, text: str) -> SentimentResult:
        """使用BERT模型分析"""
        try:
            # 编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # 解析结果
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            pred_idx = np.argmax(probs)
            polarity = label_map[pred_idx]

            # 计算极性得分 (-1 到 1)
            polarity_score = probs[2] - probs[0]

            # 获取情绪
            emotion, emotion_scores = self._detect_emotion(text)

            return SentimentResult(
                text=text,
                polarity=polarity,
                polarity_score=float(polarity_score),
                emotion=emotion,
                emotion_scores=emotion_scores,
                intensity=float(max(probs)),
                confidence=float(probs[pred_idx])
            )

        except Exception as e:
            logger.error(f"BERT分析失败: {e}")
            return self._analyze_with_dictionary(text)

    def _analyze_with_snownlp(self, text: str) -> SentimentResult:
        """使用SnowNLP分析"""
        try:
            s = self.snownlp_class(text)
            sentiment_score = s.sentiments  # 0-1, 1表示正面

            # 转换为极性
            if sentiment_score > 0.6:
                polarity = "positive"
            elif sentiment_score < 0.4:
                polarity = "negative"
            else:
                polarity = "neutral"

            # 转换为-1到1的分数
            polarity_score = (sentiment_score - 0.5) * 2

            # 获取情绪
            emotion, emotion_scores = self._detect_emotion(text)

            # 强度
            intensity = abs(polarity_score)

            return SentimentResult(
                text=text,
                polarity=polarity,
                polarity_score=float(polarity_score),
                emotion=emotion,
                emotion_scores=emotion_scores,
                intensity=float(intensity),
                confidence=float(max(sentiment_score, 1 - sentiment_score))
            )

        except Exception as e:
            logger.error(f"SnowNLP分析失败: {e}")
            return self._analyze_with_dictionary(text)

    def _analyze_with_dictionary(self, text: str) -> SentimentResult:
        """使用词典分析"""
        # 分词（简单实现）
        import jieba
        words = list(jieba.cut(text))

        # 计算情感得分
        score = 0.0
        word_count = 0
        degree = 1.0
        negation = False

        for i, word in enumerate(words):
            # 检查程度副词
            if word in self.degree_words:
                degree = self.degree_words[word]
                continue

            # 检查否定词
            if word in self.negation_words:
                negation = True
                continue

            # 检查情感词
            word_score = 0.0
            if word in self.positive_words:
                word_score = self.positive_words[word]
            elif word in self.negative_words:
                word_score = self.negative_words[word]

            if word_score != 0:
                if negation:
                    word_score = -word_score * 0.5
                word_score *= degree
                score += word_score
                word_count += 1

                # 重置
                degree = 1.0
                negation = False

        # 归一化
        if word_count > 0:
            avg_score = score / word_count
        else:
            avg_score = 0.0

        # 限制范围
        polarity_score = max(-1, min(1, avg_score))

        # 确定极性
        if polarity_score > 0.2:
            polarity = "positive"
        elif polarity_score < -0.2:
            polarity = "negative"
        else:
            polarity = "neutral"

        # 获取情绪
        emotion, emotion_scores = self._detect_emotion(text)

        return SentimentResult(
            text=text,
            polarity=polarity,
            polarity_score=float(polarity_score),
            emotion=emotion,
            emotion_scores=emotion_scores,
            intensity=float(abs(polarity_score)),
            confidence=0.7  # 词典方法固定置信度
        )

    def _detect_emotion(self, text: str) -> Tuple[str, Dict[str, float]]:
        """检测情绪类型"""
        import jieba
        words = set(jieba.cut(text))

        emotion_scores = {}

        for emotion, keywords in self.emotion_words.items():
            matches = len(words & keywords)
            emotion_scores[emotion] = matches / max(len(words), 1)

        # 添加neutral
        emotion_scores['neutral'] = 1.0 - sum(emotion_scores.values())
        emotion_scores['neutral'] = max(0, emotion_scores['neutral'])

        # 归一化
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}

        # 主要情绪
        main_emotion = max(emotion_scores, key=emotion_scores.get)

        return main_emotion, emotion_scores

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        批量分析文本

        Args:
            texts: 文本列表

        Returns:
            SentimentResult列表
        """
        results = []

        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content"
    ) -> pd.DataFrame:
        """
        分析DataFrame中的文本

        Args:
            df: 输入DataFrame
            text_column: 文本列名

        Returns:
            添加情感分析结果的DataFrame
        """
        df = df.copy()

        results = self.analyze_batch(df[text_column].fillna("").tolist())

        df["sentiment_polarity"] = [r.polarity for r in results]
        df["sentiment_score"] = [r.polarity_score for r in results]
        df["emotion"] = [r.emotion for r in results]
        df["sentiment_intensity"] = [r.intensity for r in results]
        df["sentiment_confidence"] = [r.confidence for r in results]

        logger.info(f"情感分析完成，处理 {len(df)} 条数据")

        return df

    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict:
        """
        获取情感分析摘要

        Args:
            results: SentimentResult列表

        Returns:
            摘要统计字典
        """
        if not results:
            return {}

        polarities = [r.polarity for r in results]
        scores = [r.polarity_score for r in results]
        emotions = [r.emotion for r in results]
        intensities = [r.intensity for r in results]

        return {
            "total_count": len(results),
            "polarity_distribution": {
                "positive": polarities.count("positive"),
                "neutral": polarities.count("neutral"),
                "negative": polarities.count("negative")
            },
            "avg_polarity_score": float(np.mean(scores)),
            "dominant_emotion": max(set(emotions), key=emotions.count),
            "emotion_distribution": {e: emotions.count(e) for e in set(emotions)},
            "avg_intensity": float(np.mean(intensities))
        }


if __name__ == "__main__":
    # 测试
    analyzer = SentimentAnalyzer(model_type="snownlp")

    test_texts = [
        "今天心情真好，天气也很棒！",
        "太糟糕了，这个服务让我很失望。",
        "明天要开会，需要准备材料。",
        "疫情让人很担心，希望早日结束。",
        "非常满意这次的购物体验，强烈推荐！"
    ]

    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\n文本: {text}")
        print(f"极性: {result.polarity} (得分: {result.polarity_score:.3f})")
        print(f"情绪: {result.emotion}")
        print(f"强度: {result.intensity:.3f}")

