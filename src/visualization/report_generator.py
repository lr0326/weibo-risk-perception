"""
报告生成模块
生成分析报告
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import pandas as pd
from jinja2 import Template
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


class ReportGenerator:
    """
    报告生成器

    功能：
    - HTML报告生成
    - Markdown报告生成
    - 数据摘要生成
    - 可视化图表嵌入
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化报告生成器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 报告配置
        report_config = self.config.get("visualization", {}).get("reports", {})
        self.output_format = report_config.get("output_format", "html")
        self.template = report_config.get("template", "default")

        # 输出路径
        self.output_path = self.config.get("paths", {}).get("data", {}).get("outputs", "data/outputs")

        logger.info("报告生成器初始化完成")

    def generate_report(
        self,
        results: Dict,
        output_path: str = None,
        format: str = None,
        title: str = "舆情分析报告"
    ) -> str:
        """
        生成分析报告

        Args:
            results: 分析结果字典
            output_path: 输出路径
            format: 输出格式
            title: 报告标题

        Returns:
            报告文件路径
        """
        if format is None:
            format = self.output_format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.{format}"
            output_path = os.path.join(self.output_path, filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if format == "html":
            content = self._generate_html_report(results, title)
        elif format == "markdown" or format == "md":
            content = self._generate_markdown_report(results, title)
        elif format == "json":
            content = json.dumps(results, ensure_ascii=False, indent=2, default=str)
        else:
            raise ValueError(f"不支持的格式: {format}")

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"报告已生成: {output_path}")

        return output_path

    def _generate_html_report(self, results: Dict, title: str) -> str:
        """生成HTML报告"""
        template = Template(self._get_html_template())

        # 准备数据
        context = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            **self._prepare_report_data(results)
        }

        return template.render(**context)

    def _generate_markdown_report(self, results: Dict, title: str) -> str:
        """生成Markdown报告"""
        lines = [
            f"# {title}",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n"
        ]

        # 概览
        lines.append("## 📊 分析概览\n")

        if "sample_size" in results:
            lines.append(f"- **样本量**: {results['sample_size']}")
        if "risk_level" in results:
            lines.append(f"- **风险等级**: {results['risk_level']}")
        if "risk_score" in results:
            lines.append(f"- **风险得分**: {results['risk_score']:.1f}")

        lines.append("")

        # 情感分析
        if "sentiment_summary" in results:
            lines.append("## 😊 情感分析\n")
            sentiment = results["sentiment_summary"]

            if "avg_polarity" in sentiment:
                lines.append(f"- **平均极性**: {sentiment['avg_polarity']:.3f}")
            if "dominant_emotion" in sentiment:
                lines.append(f"- **主要情绪**: {sentiment['dominant_emotion']}")
            if "emotion_distribution" in sentiment:
                lines.append("\n### 情绪分布\n")
                for emotion, count in sentiment["emotion_distribution"].items():
                    lines.append(f"- {emotion}: {count}")

            lines.append("")

        # 风险维度
        if "dimension_scores" in results:
            lines.append("## ⚠️ 风险维度分析\n")
            dimension_names = {
                "health_risk": "健康风险",
                "economic_risk": "经济风险",
                "social_risk": "社会风险",
                "political_risk": "政治风险"
            }
            for dim, score in results["dimension_scores"].items():
                name = dimension_names.get(dim, dim)
                lines.append(f"- **{name}**: {score:.1f}")
            lines.append("")

        # 预警信息
        if "warnings" in results and results["warnings"]:
            lines.append("## 🚨 预警信息\n")
            for warning in results["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")

        # 建议
        if "recommendations" in results and results["recommendations"]:
            lines.append("## 💡 应对建议\n")
            for i, rec in enumerate(results["recommendations"], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # 趋势预测
        if "predictions" in results:
            lines.append("## 🔮 趋势预测\n")
            lines.append("未来趋势预测数据已包含在详细结果中。")
            lines.append("")

        # 结语
        lines.append("---\n")
        lines.append("*本报告由微博舆情风险监控系统自动生成*")

        return "\n".join(lines)

    def _prepare_report_data(self, results: Dict) -> Dict:
        """准备报告数据"""
        data = {
            "risk_level": results.get("risk_level", "未知"),
            "risk_score": results.get("risk_score", 0),
            "sample_size": results.get("sample_size", 0),
            "warnings": results.get("warnings", []),
            "recommendations": results.get("recommendations", []),
            "sentiment_summary": results.get("sentiment_summary", {}),
            "dimension_scores": results.get("dimension_scores", {}),
            "trend": results.get("trend", "stable")
        }

        # 风险等级颜色
        risk_colors = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545"
        }
        data["risk_color"] = risk_colors.get(str(data["risk_level"]).lower(), "#6c757d")

        # 趋势图标
        trend_icons = {
            "rising": "📈",
            "stable": "➡️",
            "declining": "📉"
        }
        data["trend_icon"] = trend_icons.get(data["trend"], "➡️")

        return data

    def _get_html_template(self) -> str:
        """获取HTML模板"""
        return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Microsoft YaHei', sans-serif; background: #f8f9fa; }
        .report-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 0; }
        .metric-card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .risk-badge { font-size: 2em; padding: 10px 30px; border-radius: 50px; }
        .warning-item { padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; background: #fff3cd; border-radius: 5px; }
        .recommendation-item { padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; background: #d4edda; border-radius: 5px; }
        .dimension-bar { height: 25px; border-radius: 5px; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="report-header text-center">
        <div class="container">
            <h1>{{ title }}</h1>
            <p class="lead">生成时间: {{ generated_at }}</p>
        </div>
    </div>
    
    <div class="container py-5">
        <!-- 风险概览 -->
        <div class="row mb-4">
            <div class="col-md-4">
                <div class="metric-card text-center">
                    <h5 class="text-muted">风险等级</h5>
                    <span class="risk-badge" style="background: {{ risk_color }}; color: white;">
                        {{ risk_level }}
                    </span>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card text-center">
                    <h5 class="text-muted">风险得分</h5>
                    <h2>{{ "%.1f"|format(risk_score) }}</h2>
                    <small class="text-muted">/ 100</small>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card text-center">
                    <h5 class="text-muted">趋势</h5>
                    <h2>{{ trend_icon }} {{ trend }}</h2>
                </div>
            </div>
        </div>
        
        <!-- 情感分析 -->
        {% if sentiment_summary %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h4>😊 情感分析摘要</h4>
                    <hr>
                    <div class="row">
                        <div class="col-md-4">
                            <p><strong>平均极性:</strong> {{ "%.3f"|format(sentiment_summary.get('avg_polarity', 0)) }}</p>
                        </div>
                        <div class="col-md-4">
                            <p><strong>主要情绪:</strong> {{ sentiment_summary.get('dominant_emotion', '-') }}</p>
                        </div>
                        <div class="col-md-4">
                            <p><strong>平均强度:</strong> {{ "%.3f"|format(sentiment_summary.get('avg_intensity', 0)) }}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
        
        <!-- 风险维度 -->
        {% if dimension_scores %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h4>⚠️ 风险维度分析</h4>
                    <hr>
                    {% for dim, score in dimension_scores.items() %}
                    <div class="mb-3">
                        <div class="d-flex justify-content-between">
                            <span>{{ dim }}</span>
                            <span>{{ "%.1f"|format(score) }}</span>
                        </div>
                        <div class="progress" style="height: 25px;">
                            <div class="progress-bar {% if score > 70 %}bg-danger{% elif score > 50 %}bg-warning{% else %}bg-success{% endif %}" 
                                 style="width: {{ score }}%"></div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}
        
        <!-- 预警信息 -->
        {% if warnings %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h4>🚨 预警信息</h4>
                    <hr>
                    {% for warning in warnings %}
                    <div class="warning-item">{{ warning }}</div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}
        
        <!-- 建议 -->
        {% if recommendations %}
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h4>💡 应对建议</h4>
                    <hr>
                    {% for rec in recommendations %}
                    <div class="recommendation-item">{{ rec }}</div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}
        
        <div class="text-center text-muted mt-5">
            <p>本报告由微博舆情风险监控系统自动生成</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """
        生成数据摘要

        Args:
            df: 数据DataFrame

        Returns:
            摘要字典
        """
        summary = {
            "total_records": len(df),
            "time_range": {},
            "sentiment_stats": {},
            "engagement_stats": {},
            "user_stats": {}
        }

        # 时间范围
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            summary["time_range"] = {
                "start": str(df["created_at"].min()),
                "end": str(df["created_at"].max())
            }

        # 情感统计
        if "sentiment_score" in df.columns:
            summary["sentiment_stats"] = {
                "mean": float(df["sentiment_score"].mean()),
                "std": float(df["sentiment_score"].std()),
                "min": float(df["sentiment_score"].min()),
                "max": float(df["sentiment_score"].max())
            }

        if "sentiment_polarity" in df.columns:
            dist = df["sentiment_polarity"].value_counts(normalize=True).to_dict()
            summary["sentiment_stats"]["distribution"] = {
                k: float(v) for k, v in dist.items()
            }

        # 互动统计
        engagement_cols = ["reposts_count", "comments_count", "attitudes_count"]
        for col in engagement_cols:
            if col in df.columns:
                summary["engagement_stats"][col] = {
                    "total": int(df[col].sum()),
                    "mean": float(df[col].mean()),
                    "max": int(df[col].max())
                }

        # 用户统计
        if "user_id" in df.columns:
            summary["user_stats"] = {
                "unique_users": int(df["user_id"].nunique()),
                "avg_posts_per_user": float(len(df) / df["user_id"].nunique())
            }

        return summary


if __name__ == "__main__":
    # 测试
    generator = ReportGenerator()

    # 模拟结果
    results = {
        "sample_size": 1000,
        "risk_level": "medium",
        "risk_score": 55.5,
        "trend": "rising",
        "sentiment_summary": {
            "avg_polarity": 0.15,
            "dominant_emotion": "neutral",
            "avg_intensity": 0.45,
            "emotion_distribution": {"joy": 200, "neutral": 500, "anger": 150, "fear": 100, "sadness": 50}
        },
        "dimension_scores": {
            "health_risk": 45.0,
            "economic_risk": 60.0,
            "social_risk": 55.0,
            "political_risk": 35.0
        },
        "warnings": [
            "⚠️ 中风险提示：当前风险指数为55.5，建议密切监控",
            "⚠️ 经济风险维度得分较高 (60.0)"
        ],
        "recommendations": [
            "建议持续关注舆情发展态势",
            "建议准备风险应对预案",
            "建议关注经济相关诉求"
        ]
    }

    # 生成报告
    html_path = generator.generate_report(results, format="html")
    md_path = generator.generate_report(results, format="markdown")

    print(f"HTML报告: {html_path}")
    print(f"Markdown报告: {md_path}")

