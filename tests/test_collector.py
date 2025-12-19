"""
测试数据采集模块
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.weibo_collector import WeiboDataCollector, MockWeiboDataGenerator


class TestMockDataGenerator:
    """测试模拟数据生成器"""

    def test_generate_mock_data(self):
        """测试生成模拟数据"""
        generator = MockWeiboDataGenerator()
        df = generator.generate_mock_data(100)

        assert len(df) == 100
        assert "weibo_id" in df.columns
        assert "content" in df.columns
        assert "user_id" in df.columns
        assert "created_at" in df.columns

    def test_mock_data_content(self):
        """测试模拟数据内容"""
        generator = MockWeiboDataGenerator()
        df = generator.generate_mock_data(50)

        # 检查内容不为空
        assert df["content"].notna().all()
        assert df["weibo_id"].nunique() == 50  # ID唯一


class TestWeiboCollector:
    """测试微博采集器"""

    def test_collector_init(self):
        """测试采集器初始化"""
        collector = WeiboDataCollector()
        assert collector is not None
        assert collector.batch_size > 0

    def test_parse_weibo_list(self):
        """测试解析微博列表"""
        collector = WeiboDataCollector()

        mock_weibos = [
            {
                "id": "123456",
                "text": "测试微博内容",
                "created_at": "Tue May 31 17:46:55 +0800 2023",
                "reposts_count": 10,
                "comments_count": 5,
                "attitudes_count": 100,
                "user": {
                    "id": "user123",
                    "screen_name": "测试用户",
                    "followers_count": 1000,
                    "verified": False
                }
            }
        ]

        df = collector._parse_weibo_list(mock_weibos)

        assert len(df) == 1
        assert df.iloc[0]["weibo_id"] == "123456"
        assert df.iloc[0]["content"] == "测试微博内容"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

