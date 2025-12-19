"""
文本清洗模块
提供中文文本的清洗、标准化等预处理功能
"""

import re
import unicodedata
from typing import List, Optional, Set

import jieba
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


class TextCleaner:
    """
    文本清洗器

    支持功能：
    - URL移除
    - @提及移除
    - 话题标签处理
    - 表情符号处理
    - 中文分词
    - 停用词过滤
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化文本清洗器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 清洗配置
        clean_config = self.config.get("preprocessing", {}).get("text_cleaning", {})
        self.remove_urls = clean_config.get("remove_urls", True)
        self.remove_mentions = clean_config.get("remove_mentions", True)
        self.remove_hashtags = clean_config.get("remove_hashtags", False)
        self.remove_emojis = clean_config.get("remove_emojis", False)
        self.remove_punctuation = clean_config.get("remove_punctuation", False)
        self.convert_lowercase = clean_config.get("convert_lowercase", False)
        self.min_length = clean_config.get("min_length", 5)

        # 分词配置
        token_config = self.config.get("preprocessing", {}).get("tokenization", {})
        self.tokenizer_engine = token_config.get("engine", "jieba")
        user_dict_path = token_config.get("user_dict", "")
        stop_words_path = token_config.get("stop_words", "")

        # 加载用户词典
        if user_dict_path:
            self._load_user_dict(user_dict_path)

        # 加载停用词
        self.stop_words: Set[str] = set()
        if stop_words_path:
            self._load_stop_words(stop_words_path)
        else:
            self._load_default_stop_words()

        # 编译正则表达式
        self._compile_patterns()

        logger.info("文本清洗器初始化完成")

    def _compile_patterns(self):
        """编译正则表达式模式"""
        # URL模式
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+'
            r'|www\.[^\s<>"{}|\\^`\[\]]+'
        )

        # @提及模式
        self.mention_pattern = re.compile(r'@[\w\u4e00-\u9fff]+')

        # 话题标签模式 (#话题#)
        self.hashtag_pattern = re.compile(r'#[^#]+#')

        # 表情符号模式 [表情]
        self.emoji_pattern = re.compile(r'\[[\w\u4e00-\u9fff]+\]')

        # 多余空白字符
        self.whitespace_pattern = re.compile(r'\s+')

        # 中文标点符号
        self.chinese_punctuation = re.compile(
            r'[，。！？、；：""''【】《》（）…—～·]'
        )

        # 非中文英文数字字符
        self.invalid_chars = re.compile(
            r'[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a\u0030-\u0039\s]'
        )

    def _load_user_dict(self, path: str):
        """加载用户自定义词典"""
        try:
            jieba.load_userdict(path)
            logger.info(f"加载用户词典: {path}")
        except FileNotFoundError:
            logger.warning(f"用户词典不存在: {path}")
        except Exception as e:
            logger.error(f"加载用户词典失败: {e}")

    def _load_stop_words(self, path: str):
        """加载停用词表"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.stop_words = set(line.strip() for line in f if line.strip())
            logger.info(f"加载停用词 {len(self.stop_words)} 个")
        except FileNotFoundError:
            logger.warning(f"停用词文件不存在: {path}")
            self._load_default_stop_words()
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")

    def _load_default_stop_words(self):
        """加载默认停用词"""
        default_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它',
            '们', '这个', '那个', '什么', '怎么', '为什么', '因为', '所以',
            '但是', '然后', '如果', '可以', '没', '吗', '呢', '吧', '啊',
            '哦', '嗯', '呀', '哈', '哎', '唉', '嘿', '喂', '诶'
        }
        self.stop_words = default_stop_words
        logger.info(f"使用默认停用词 {len(self.stop_words)} 个")

    def clean(self, text: str) -> str:
        """
        清洗单条文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""

        # 移除URL
        if self.remove_urls:
            text = self.url_pattern.sub('', text)

        # 移除@提及
        if self.remove_mentions:
            text = self.mention_pattern.sub('', text)

        # 处理话题标签
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub('', text)
        else:
            # 保留话题内容，移��#符号
            text = re.sub(r'#([^#]+)#', r'\1', text)

        # 处理表情符号
        if self.remove_emojis:
            text = self.emoji_pattern.sub('', text)

        # 移除标点符号
        if self.remove_punctuation:
            text = self.chinese_punctuation.sub('', text)
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)

        # 转小写
        if self.convert_lowercase:
            text = text.lower()

        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)

        # 压缩空白字符
        text = self.whitespace_pattern.sub(' ', text)

        # 去除首尾空白
        text = text.strip()

        return text

    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """
        分词

        Args:
            text: 输入文本
            remove_stop_words: 是否移除停用词

        Returns:
            分词结果列表
        """
        if not text:
            return []

        # 先清洗
        cleaned = self.clean(text)

        # 分词
        if self.tokenizer_engine == "jieba":
            tokens = list(jieba.cut(cleaned))
        else:
            # 默认使用jieba
            tokens = list(jieba.cut(cleaned))

        # 过滤
        filtered = []
        for token in tokens:
            token = token.strip()

            # 跳过空白
            if not token:
                continue

            # 跳过过短的词
            if len(token) < 2 and not token.isdigit():
                continue

            # 移除停用词
            if remove_stop_words and token in self.stop_words:
                continue

            filtered.append(token)

        return filtered

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        output_column: str = "cleaned_content",
        tokenize: bool = False,
        token_column: str = "tokens"
    ) -> pd.DataFrame:
        """
        批量处理DataFrame中的文本

        Args:
            df: 输入DataFrame
            text_column: 文本列名
            output_column: 输出列名
            tokenize: 是否进行分词
            token_column: 分词结果列名

        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        # 清洗
        df[output_column] = df[text_column].apply(self.clean)

        # 过滤过短的文本
        df = df[df[output_column].str.len() >= self.min_length]

        # 分词
        if tokenize:
            df[token_column] = df[output_column].apply(self.tokenize)

        logger.info(f"处理完成，有效数据 {len(df)} 条")

        return df

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        method: str = "tfidf"
    ) -> List[tuple]:
        """
        提取关键词

        Args:
            text: 输入文本
            top_k: 返回关键词数量
            method: 提取方法 (tfidf, textrank)

        Returns:
            关键词列表 [(word, weight), ...]
        """
        if not text:
            return []

        cleaned = self.clean(text)

        if method == "tfidf":
            import jieba.analyse
            keywords = jieba.analyse.extract_tags(
                cleaned, topK=top_k, withWeight=True
            )
        elif method == "textrank":
            import jieba.analyse
            keywords = jieba.analyse.textrank(
                cleaned, topK=top_k, withWeight=True
            )
        else:
            keywords = []

        return keywords


if __name__ == "__main__":
    # 测试
    cleaner = TextCleaner()

    test_texts = [
        "今天天气真好！#北京生活# @小明 https://example.com [开心]",
        "疫情防控政策调整了，大家怎么看？",
        "经济形势分析：GDP增长达到预期目标 👍"
    ]

    for text in test_texts:
        print(f"\n原文: {text}")
        print(f"清洗: {cleaner.clean(text)}")
        print(f"分词: {cleaner.tokenize(text)}")

