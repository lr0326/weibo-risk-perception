"""
测试预处理模块
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_cleaner import TextCleaner


class TestTextCleaner:
    """测试文本清洗器"""

    @pytest.fixture
    def cleaner(self):
        """创建清洗器实例"""
        return TextCleaner()

    def test_clean_url(self, cleaner):
        """测试URL清洗"""
        text = "查看详情: https://example.com/path 点击链接"
        cleaned = cleaner.clean(text)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned

    def test_clean_mention(self, cleaner):
        """测试@提及清洗"""
        text = "感谢@张三 @李四的帮助"
        cleaned = cleaner.clean(text)
        assert "@张三" not in cleaned
        assert "@李四" not in cleaned

    def test_clean_hashtag(self, cleaner):
        """测试话题标签处理"""
        text = "今天#北京生活#很开心"
        cleaned = cleaner.clean(text)
        # 默认保留话题内容，移除#符号
        assert "#" not in cleaned
        assert "北京生活" in cleaned

    def test_tokenize(self, cleaner):
        """测试分词"""
        text = "我爱北京天安门"
        tokens = cleaner.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "天安门" in tokens or "北京" in tokens

    def test_tokenize_remove_stopwords(self, cleaner):
        """测试分词时移除停用词"""
        text = "我的朋友在北京工作"
        tokens_with_stop = cleaner.tokenize(text, remove_stop_words=False)
        tokens_without_stop = cleaner.tokenize(text, remove_stop_words=True)

        # 移除停用词后词数应该减少
        assert len(tokens_without_stop) <= len(tokens_with_stop)

    def test_extract_keywords(self, cleaner):
        """测试关键词提取"""
        text = "人工智能技术在医疗领域的应用越来越广泛，机器学习算法可以帮助医生进行诊断"
        keywords = cleaner.extract_keywords(text, top_k=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        if keywords:
            assert isinstance(keywords[0], tuple)
            assert len(keywords[0]) == 2  # (word, weight)

    def test_clean_empty(self, cleaner):
        """测试空文本处理"""
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""

    def test_tokenize_empty(self, cleaner):
        """测试空文本分词"""
        assert cleaner.tokenize("") == []
        assert cleaner.tokenize(None) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

