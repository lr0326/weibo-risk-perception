"""
数据采集模块
"""

from .weibo_collector import WeiboDataCollector, MockWeiboDataGenerator

try:
    from .stream_collector import StreamCollector
except ImportError:
    StreamCollector = None

__all__ = ["WeiboDataCollector", "MockWeiboDataGenerator", "StreamCollector"]

