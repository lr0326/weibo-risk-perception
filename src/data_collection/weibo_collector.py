"""
微博数据采集器
提供微博数据的采集功能，支持关键词搜索、用户微博获取、话题采集等
"""

import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from urllib.parse import urlencode

import requests
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config, retry_on_failure


class WeiboDataCollector:
    """
    微博数据采集器

    支持功能：
    - 关键词搜索微博
    - 获取用户微博列表
    - 获取热门话题
    - 获取微博评论
    - 实时搜索
    """

    def __init__(self, access_token: str = None, config_path: str = "config/config.yaml"):
        """
        初始化采集器

        Args:
            access_token: 微博API访问令牌
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.access_token = access_token

        # API配置
        api_config = self.config.get("data_collection", {}).get("weibo_api", {})
        self.base_url = api_config.get("base_url", "https://api.weibo.com/2/")
        self.rate_limit = api_config.get("rate_limit", 150)
        self.retry_times = api_config.get("retry_times", 3)
        self.retry_delay = api_config.get("retry_delay", 5)

        # 爬虫配置
        crawler_config = self.config.get("data_collection", {}).get("crawler", {})
        self.batch_size = crawler_config.get("batch_size", 100)
        self.max_pages = crawler_config.get("max_pages", 50)
        self.sleep_interval = crawler_config.get("sleep_interval", 1.0)
        self.timeout = crawler_config.get("timeout", 30)

        # 请求计数器
        self.request_count = 0
        self.last_reset_time = datetime.now()

        # 请求头
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }

        logger.info("微博数据采集器初始化完成")

    def _check_rate_limit(self):
        """检查速率限制"""
        now = datetime.now()
        if (now - self.last_reset_time).total_seconds() > 3600:
            self.request_count = 0
            self.last_reset_time = now

        if self.request_count >= self.rate_limit:
            wait_time = 3600 - (now - self.last_reset_time).total_seconds()
            if wait_time > 0:
                logger.warning(f"达到速率限制，等待 {wait_time:.0f} 秒")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_reset_time = datetime.now()

    @retry_on_failure(max_retries=3, delay=5)
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        发送API请求

        Args:
            endpoint: API端点
            params: 请求参数

        Returns:
            API响应数据
        """
        self._check_rate_limit()

        if params is None:
            params = {}

        if self.access_token:
            params["access_token"] = self.access_token

        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            self.request_count += 1

            return response.json()

        except requests.RequestException as e:
            logger.error(f"API请求失败: {e}")
            raise

    def search_weibo_by_keyword(
        self,
        keyword: str,
        count: int = 50,
        pages: int = 10,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> pd.DataFrame:
        """
        按关键词搜索微博

        Args:
            keyword: 搜索关键词
            count: 每页数量
            pages: 采集页数
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            微博数据DataFrame
        """
        logger.info(f"开始搜索关键词: {keyword}")

        all_weibos = []

        for page in range(1, min(pages, self.max_pages) + 1):
            params = {
                "q": keyword,
                "count": min(count, self.batch_size),
                "page": page
            }

            if start_time:
                params["starttime"] = int(start_time.timestamp())
            if end_time:
                params["endtime"] = int(end_time.timestamp())

            try:
                result = self._make_request("search/topics.json", params)

                if "statuses" in result:
                    weibos = result["statuses"]
                    all_weibos.extend(weibos)
                    logger.debug(f"第 {page} 页获取 {len(weibos)} 条微博")
                else:
                    logger.warning(f"第 {page} 页未获取到数据")
                    break

            except Exception as e:
                logger.error(f"采集第 {page} 页失败: {e}")
                continue

            # 请求间隔
            time.sleep(self.sleep_interval)

        if all_weibos:
            df = self._parse_weibo_list(all_weibos)
            logger.info(f"共采集 {len(df)} 条微博")
            return df
        else:
            logger.warning("未采集到任何数据")
            return pd.DataFrame()

    def get_user_weibos(
        self,
        user_id: str,
        count: int = 50,
        pages: int = 10
    ) -> pd.DataFrame:
        """
        获取用户微博列表

        Args:
            user_id: 用户ID
            count: 每页数量
            pages: 采集页数

        Returns:
            微博数据DataFrame
        """
        logger.info(f"开始获取用户 {user_id} 的微博")

        all_weibos = []

        for page in range(1, min(pages, self.max_pages) + 1):
            params = {
                "uid": user_id,
                "count": min(count, self.batch_size),
                "page": page
            }

            try:
                result = self._make_request("statuses/user_timeline.json", params)

                if "statuses" in result:
                    weibos = result["statuses"]
                    all_weibos.extend(weibos)
                else:
                    break

            except Exception as e:
                logger.error(f"采集失败: {e}")
                continue

            time.sleep(self.sleep_interval)

        if all_weibos:
            return self._parse_weibo_list(all_weibos)
        return pd.DataFrame()

    def get_hot_topics(self) -> pd.DataFrame:
        """
        获取热门话题

        Returns:
            热门话题DataFrame
        """
        logger.info("获取热门话题")

        try:
            result = self._make_request("trends/hourly.json")

            if "trends" in result:
                topics = []
                for date_key, trend_list in result["trends"].items():
                    for trend in trend_list:
                        topics.append({
                            "name": trend.get("name", ""),
                            "query": trend.get("query", ""),
                            "amount": trend.get("amount", 0),
                            "date": date_key
                        })

                return pd.DataFrame(topics)

        except Exception as e:
            logger.error(f"获取热门话题失败: {e}")

        return pd.DataFrame()

    def get_weibo_comments(
        self,
        weibo_id: str,
        count: int = 50,
        pages: int = 5
    ) -> pd.DataFrame:
        """
        获取微博评论

        Args:
            weibo_id: 微博ID
            count: 每页数量
            pages: 采集页数

        Returns:
            评论数据DataFrame
        """
        logger.info(f"获取微博 {weibo_id} 的评论")

        all_comments = []

        for page in range(1, min(pages, 10) + 1):
            params = {
                "id": weibo_id,
                "count": min(count, 50),
                "page": page
            }

            try:
                result = self._make_request("comments/show.json", params)

                if "comments" in result:
                    comments = result["comments"]
                    all_comments.extend(comments)
                else:
                    break

            except Exception as e:
                logger.error(f"采集评论失败: {e}")
                continue

            time.sleep(self.sleep_interval)

        if all_comments:
            return self._parse_comment_list(all_comments)
        return pd.DataFrame()

    def _parse_weibo_list(self, weibos: List[Dict]) -> pd.DataFrame:
        """
        解析微博列表

        Args:
            weibos: 微博数据列表

        Returns:
            解析后的DataFrame
        """
        parsed = []

        for weibo in weibos:
            try:
                user = weibo.get("user", {})

                parsed.append({
                    "weibo_id": str(weibo.get("id", "")),
                    "user_id": str(user.get("id", "")),
                    "user_name": user.get("screen_name", ""),
                    "user_followers": user.get("followers_count", 0),
                    "user_verified": user.get("verified", False),
                    "content": weibo.get("text", ""),
                    "created_at": self._parse_datetime(weibo.get("created_at", "")),
                    "reposts_count": weibo.get("reposts_count", 0),
                    "comments_count": weibo.get("comments_count", 0),
                    "attitudes_count": weibo.get("attitudes_count", 0),
                    "location": user.get("location", ""),
                    "source": weibo.get("source", ""),
                    "pic_urls": [pic.get("thumbnail_pic", "") for pic in weibo.get("pic_urls", [])],
                    "is_repost": weibo.get("retweeted_status") is not None
                })
            except Exception as e:
                logger.debug(f"解析微博失败: {e}")
                continue

        return pd.DataFrame(parsed)

    def _parse_comment_list(self, comments: List[Dict]) -> pd.DataFrame:
        """
        解析评论列表

        Args:
            comments: 评论数据列表

        Returns:
            解析后的DataFrame
        """
        parsed = []

        for comment in comments:
            try:
                user = comment.get("user", {})

                parsed.append({
                    "comment_id": str(comment.get("id", "")),
                    "weibo_id": str(comment.get("status", {}).get("id", "")),
                    "user_id": str(user.get("id", "")),
                    "user_name": user.get("screen_name", ""),
                    "content": comment.get("text", ""),
                    "created_at": self._parse_datetime(comment.get("created_at", "")),
                    "like_count": comment.get("like_count", 0)
                })
            except Exception as e:
                logger.debug(f"解析评论失败: {e}")
                continue

        return pd.DataFrame(parsed)

    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """解析微博时间格式"""
        if not date_str:
            return None

        try:
            # 微博时间格式: "Tue May 31 17:46:55 +0800 2023"
            return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            try:
                return datetime.fromisoformat(date_str)
            except ValueError:
                return None

    def save_data(self, df: pd.DataFrame, filepath: str, format: str = "csv"):
        """
        保存数据

        Args:
            df: 数据DataFrame
            filepath: 保存路径
            format: 文件格式 (csv, json, excel)
        """
        if df.empty:
            logger.warning("数据为空，跳过保存")
            return

        try:
            if format == "csv":
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
            elif format == "json":
                df.to_json(filepath, orient="records", force_ascii=False, indent=2)
            elif format == "excel":
                df.to_excel(filepath, index=False)
            else:
                raise ValueError(f"不支持的格式: {format}")

            logger.info(f"数据已保存至 {filepath}")

        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise


# 模拟数据生成器（用于测试和演示）
class MockWeiboDataGenerator:
    """
    模拟微博数据生成器
    用于在没有API访问权限时进行测试和演示
    """

    def __init__(self):
        self.emotions = ["积极", "中性", "消极"]
        self.topics = ["疫情防控", "经济发展", "环境保护", "教育改革", "社会民生"]

    def generate_mock_data(self, count: int = 100) -> pd.DataFrame:
        """
        生成模拟微博数据

        Args:
            count: 生成数量

        Returns:
            模拟数据DataFrame
        """
        import random

        data = []
        base_time = datetime.now()

        for i in range(count):
            topic = random.choice(self.topics)
            emotion = random.choice(self.emotions)

            # 根据情感生成内容
            if emotion == "积极":
                content = f"关于{topic}，我觉得很好，支持这样的政策！"
            elif emotion == "消极":
                content = f"对于{topic}问题，我表示担忧和不满"
            else:
                content = f"关于{topic}，发表一些个人看法"

            data.append({
                "weibo_id": f"mock_{i}_{hashlib.md5(str(i).encode()).hexdigest()[:8]}",
                "user_id": f"user_{random.randint(10000, 99999)}",
                "user_name": f"用户{random.randint(1, 1000)}",
                "user_followers": random.randint(100, 100000),
                "user_verified": random.random() > 0.9,
                "content": content,
                "created_at": base_time - timedelta(hours=random.randint(0, 168)),
                "reposts_count": random.randint(0, 1000),
                "comments_count": random.randint(0, 500),
                "attitudes_count": random.randint(0, 5000),
                "location": random.choice(["北京", "上海", "广州", "深圳", "成都", "杭州"]),
                "source": random.choice(["iPhone客户端", "Android客户端", "微博网页版"]),
                "pic_urls": [],
                "is_repost": random.random() > 0.7
            })

        return pd.DataFrame(data)


if __name__ == "__main__":
    # 测试模拟数据生成
    generator = MockWeiboDataGenerator()
    df = generator.generate_mock_data(50)
    print(df.head())
    print(f"\n生成 {len(df)} 条模拟数据")

