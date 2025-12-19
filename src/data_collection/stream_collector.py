"""
实时流数据采集器
支持微博实时数据的流式采集
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
from queue import Queue
from threading import Thread, Event

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


class StreamCollector:
    """
    实时流数据采集器

    支持功能：
    - 实时关键词监听
    - 数据缓冲与批量处理
    - 断线重连
    """

    def __init__(
        self,
        access_token: str = None,
        config_path: str = "config/config.yaml"
    ):
        """
        初始化流采集器

        Args:
            access_token: 微博API访问令牌
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.access_token = access_token

        # 流配置
        stream_config = self.config.get("data_collection", {}).get("stream", {})
        self.enabled = stream_config.get("enabled", False)
        self.buffer_size = stream_config.get("buffer_size", 1000)
        self.flush_interval = stream_config.get("flush_interval", 60)

        # 数据缓冲区
        self.buffer: Queue = Queue(maxsize=self.buffer_size)

        # 控制标志
        self._stop_event = Event()
        self._is_running = False

        # 回调函数
        self._callbacks: List[Callable] = []

        logger.info("流数据采集器初始化完成")

    def register_callback(self, callback: Callable[[Dict], None]):
        """
        注册数据处理回调函数

        Args:
            callback: 回调函数，接收单条数据
        """
        self._callbacks.append(callback)
        logger.debug(f"注册回调函数: {callback.__name__}")

    def start(self, keywords: List[str]):
        """
        启动流采集

        Args:
            keywords: 监听关键词列表
        """
        if self._is_running:
            logger.warning("流采集器已在运行")
            return

        self._stop_event.clear()
        self._is_running = True

        # 启动采集线程
        collector_thread = Thread(
            target=self._collect_loop,
            args=(keywords,),
            daemon=True
        )
        collector_thread.start()

        # 启动处理线程
        processor_thread = Thread(
            target=self._process_loop,
            daemon=True
        )
        processor_thread.start()

        logger.info(f"流采集已启动，监听关键词: {keywords}")

    def stop(self):
        """停止流采集"""
        self._stop_event.set()
        self._is_running = False
        logger.info("流采集已停止")

    def _collect_loop(self, keywords: List[str]):
        """采集循环（模拟实现）"""
        import random

        while not self._stop_event.is_set():
            try:
                # 模拟数据采集
                # 实际应用中这里会连接到微博流API
                mock_data = self._generate_mock_stream_data(keywords)

                if mock_data and not self.buffer.full():
                    self.buffer.put(mock_data)

                time.sleep(random.uniform(0.5, 2.0))

            except Exception as e:
                logger.error(f"采集异常: {e}")
                time.sleep(5)

    def _process_loop(self):
        """处理循环"""
        batch = []
        last_flush = time.time()

        while not self._stop_event.is_set() or not self.buffer.empty():
            try:
                # 非阻塞获取
                if not self.buffer.empty():
                    data = self.buffer.get(timeout=1)
                    batch.append(data)

                    # 调用回调
                    for callback in self._callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"回调执行失败: {e}")

                # 定时刷新
                if time.time() - last_flush >= self.flush_interval:
                    if batch:
                        self._flush_batch(batch)
                        batch = []
                    last_flush = time.time()

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"处理异常: {e}")

    def _flush_batch(self, batch: List[Dict]):
        """刷新批次数据"""
        logger.info(f"刷新批次数据，共 {len(batch)} 条")
        # 这里可以添加批量保存逻辑

    def _generate_mock_stream_data(self, keywords: List[str]) -> Optional[Dict]:
        """生成模拟流数据"""
        import random
        import hashlib

        keyword = random.choice(keywords)

        return {
            "weibo_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            "content": f"关于{keyword}的实时讨论内容",
            "created_at": datetime.now().isoformat(),
            "keyword": keyword,
            "reposts_count": random.randint(0, 100),
            "comments_count": random.randint(0, 50),
            "attitudes_count": random.randint(0, 500)
        }

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._is_running

    @property
    def buffer_count(self) -> int:
        """缓冲区数据量"""
        return self.buffer.qsize()


class AsyncStreamCollector:
    """
    异步流数据采集器
    使用asyncio实现高性能采集
    """

    def __init__(self, access_token: str = None):
        self.access_token = access_token
        self._is_running = False
        self.data_queue: asyncio.Queue = asyncio.Queue()

    async def start_async(self, keywords: List[str]):
        """异步启动采集"""
        self._is_running = True

        tasks = [
            asyncio.create_task(self._collect_task(keywords)),
            asyncio.create_task(self._process_task())
        ]

        await asyncio.gather(*tasks)

    async def _collect_task(self, keywords: List[str]):
        """异步采集任务"""
        import random

        while self._is_running:
            # 模拟异步采集
            await asyncio.sleep(random.uniform(0.5, 2.0))

            mock_data = {
                "keyword": random.choice(keywords),
                "content": "异步采集的数据",
                "timestamp": datetime.now().isoformat()
            }

            await self.data_queue.put(mock_data)

    async def _process_task(self):
        """异步处理任务"""
        while self._is_running:
            try:
                data = await asyncio.wait_for(
                    self.data_queue.get(),
                    timeout=5.0
                )
                logger.debug(f"处理数据: {data}")
            except asyncio.TimeoutError:
                continue

    def stop(self):
        """停止采集"""
        self._is_running = False


if __name__ == "__main__":
    # 测试流采集器
    collector = StreamCollector()

    def sample_callback(data):
        print(f"收到数据: {data['content'][:50]}...")

    collector.register_callback(sample_callback)

    try:
        collector.start(["疫情", "经济", "教育"])
        time.sleep(10)
    finally:
        collector.stop()

