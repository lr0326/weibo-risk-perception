"""
可视化模块
"""

try:
    from .dashboard import Dashboard
except ImportError:
    Dashboard = None

try:
    from .report_generator import ReportGenerator
except ImportError:
    ReportGenerator = None

__all__ = ["Dashboard", "ReportGenerator"]

