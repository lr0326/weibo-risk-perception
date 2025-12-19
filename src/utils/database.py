"""
数据库操作模块
支持MongoDB和Redis
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


class DatabaseManager:
    """
    数据库管理器

    支持：
    - MongoDB: 主数据存储
    - Redis: 缓存和实时数据
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化数据库管理器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        db_config = self.config.get("database", {})

        # MongoDB配置
        mongo_config = db_config.get("mongodb", {})
        self.mongo_host = mongo_config.get("host", "localhost")
        self.mongo_port = mongo_config.get("port", 27017)
        self.mongo_db = mongo_config.get("database", "weibo_risk")
        self.mongo_username = mongo_config.get("username", "")
        self.mongo_password = mongo_config.get("password", "")

        # Redis配置
        redis_config = db_config.get("redis", {})
        self.redis_host = redis_config.get("host", "localhost")
        self.redis_port = redis_config.get("port", 6379)
        self.redis_db = redis_config.get("db", 0)
        self.redis_password = redis_config.get("password", "")

        # 连接对象
        self.mongo_client = None
        self.mongo_database = None
        self.redis_client = None

        logger.info("数据库管理器初始化完成")

    def connect_mongodb(self):
        """连接MongoDB"""
        try:
            from pymongo import MongoClient

            if self.mongo_username and self.mongo_password:
                uri = f"mongodb://{self.mongo_username}:{self.mongo_password}@{self.mongo_host}:{self.mongo_port}"
            else:
                uri = f"mongodb://{self.mongo_host}:{self.mongo_port}"

            self.mongo_client = MongoClient(uri)
            self.mongo_database = self.mongo_client[self.mongo_db]

            # 测试连接
            self.mongo_client.admin.command('ping')

            logger.info(f"MongoDB连接成功: {self.mongo_host}:{self.mongo_port}/{self.mongo_db}")

        except ImportError:
            logger.warning("pymongo未安装，MongoDB功能不可用")
        except Exception as e:
            logger.error(f"MongoDB连接失败: {e}")

    def connect_redis(self):
        """连接Redis"""
        try:
            import redis

            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password if self.redis_password else None,
                decode_responses=True
            )

            # 测试连接
            self.redis_client.ping()

            logger.info(f"Redis连接成功: {self.redis_host}:{self.redis_port}")

        except ImportError:
            logger.warning("redis未安装，Redis功能不可用")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")

    def connect_all(self):
        """连接所有数据库"""
        self.connect_mongodb()
        self.connect_redis()

    # ========== MongoDB操作 ==========

    def insert_weibos(self, weibos: List[Dict], collection: str = "weibos") -> int:
        """
        批量插入微博数据

        Args:
            weibos: 微博数据列表
            collection: 集合名称

        Returns:
            插入数量
        """
        if self.mongo_database is None:
            logger.error("MongoDB未连接")
            return 0

        if not weibos:
            return 0

        try:
            coll = self.mongo_database[collection]
            result = coll.insert_many(weibos)
            logger.info(f"插入 {len(result.inserted_ids)} 条微博数据")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            return 0

    def find_weibos(
        self,
        query: Dict = None,
        collection: str = "weibos",
        limit: int = 1000,
        sort_by: str = "created_at",
        ascending: bool = False
    ) -> List[Dict]:
        """
        查询微博数据

        Args:
            query: 查询条件
            collection: 集合名称
            limit: 限制数量
            sort_by: 排序字段
            ascending: 是否升序

        Returns:
            微博数据列表
        """
        if self.mongo_database is None:
            logger.error("MongoDB未连接")
            return []

        if query is None:
            query = {}

        try:
            coll = self.mongo_database[collection]
            cursor = coll.find(query).sort(
                sort_by, 1 if ascending else -1
            ).limit(limit)

            return list(cursor)
        except Exception as e:
            logger.error(f"查询数据失败: {e}")
            return []

    def update_weibo(
        self,
        weibo_id: str,
        update_data: Dict,
        collection: str = "weibos"
    ) -> bool:
        """
        更新微博数据

        Args:
            weibo_id: 微博ID
            update_data: 更新数据
            collection: 集合名称

        Returns:
            是否成功
        """
        if self.mongo_database is None:
            return False

        try:
            coll = self.mongo_database[collection]
            result = coll.update_one(
                {"weibo_id": weibo_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"更新数据失败: {e}")
            return False

    def count_weibos(
        self,
        query: Dict = None,
        collection: str = "weibos"
    ) -> int:
        """
        统计微博数量

        Args:
            query: 查询条件
            collection: 集合名称

        Returns:
            数量
        """
        if self.mongo_database is None:
            return 0

        if query is None:
            query = {}

        try:
            coll = self.mongo_database[collection]
            return coll.count_documents(query)
        except Exception as e:
            logger.error(f"统计数据失败: {e}")
            return 0

    def aggregate_weibos(
        self,
        pipeline: List[Dict],
        collection: str = "weibos"
    ) -> List[Dict]:
        """
        聚合查询

        Args:
            pipeline: 聚合管道
            collection: 集合名称

        Returns:
            聚合结果
        """
        if self.mongo_database is None:
            return []

        try:
            coll = self.mongo_database[collection]
            return list(coll.aggregate(pipeline))
        except Exception as e:
            logger.error(f"聚合查询失败: {e}")
            return []

    # ========== Redis操作 ==========

    def cache_set(self, key: str, value: Any, expire: int = 3600):
        """
        设置缓存

        Args:
            key: 键
            value: 值
            expire: 过期时间（秒）
        """
        if self.redis_client is None:
            logger.warning("Redis未连接")
            return

        try:
            import json

            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, default=str)

            self.redis_client.setex(key, expire, value)
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")

    def cache_get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 键

        Returns:
            缓存值
        """
        if self.redis_client is None:
            return None

        try:
            import json

            value = self.redis_client.get(key)

            if value:
                try:
                    return json.loads(value)
                except:
                    return value
            return None
        except Exception as e:
            logger.error(f"缓存获取失败: {e}")
            return None

    def cache_delete(self, key: str):
        """删除缓存"""
        if self.redis_client is None:
            return

        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"缓存删除失败: {e}")

    def publish_message(self, channel: str, message: str):
        """
        发布消息

        Args:
            channel: 频道
            message: 消息
        """
        if self.redis_client is None:
            return

        try:
            self.redis_client.publish(channel, message)
        except Exception as e:
            logger.error(f"消息发布失败: {e}")

    def increment_counter(self, key: str, amount: int = 1) -> int:
        """
        增加计数器

        Args:
            key: 键
            amount: 增量

        Returns:
            新值
        """
        if self.redis_client is None:
            return 0

        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"计数器增加失败: {e}")
            return 0

    # ========== 工具方法 ==========

    def to_dataframe(
        self,
        collection: str = "weibos",
        query: Dict = None,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        导出为DataFrame

        Args:
            collection: 集合名称
            query: 查询条件
            limit: 限制数量

        Returns:
            DataFrame
        """
        data = self.find_weibos(query=query, collection=collection, limit=limit)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # 移除MongoDB的_id字段
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)

        return df

    def close(self):
        """关闭所有连接"""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB连接已关闭")

        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis连接已关闭")


if __name__ == "__main__":
    # 测试
    db = DatabaseManager()

    # 测试连接（如果服务可用）
    try:
        db.connect_mongodb()
        print(f"微博数量: {db.count_weibos()}")
    except Exception as e:
        print(f"MongoDB测试跳过: {e}")

    try:
        db.connect_redis()
        db.cache_set("test_key", {"value": 123})
        print(f"缓存测试: {db.cache_get('test_key')}")
    except Exception as e:
        print(f"Redis测试跳过: {e}")

    db.close()

