"""
可视化仪表盘
基于Dash构建的交互式数据分析仪表盘
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


class Dashboard:
    """
    可视化仪表盘

    功能：
    - 实时数据展示
    - 情感趋势图表
    - 风险指数监控
    - 词云展示
    - 网络图可视化
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化仪表盘

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 可视化配置
        viz_config = self.config.get("visualization", {})
        dashboard_config = viz_config.get("dashboard", {})

        self.host = dashboard_config.get("host", "0.0.0.0")
        self.port = dashboard_config.get("port", 8050)
        self.debug = dashboard_config.get("debug", False)
        self.update_interval = dashboard_config.get("update_interval", 300) * 1000  # 转换为毫秒

        # 图表配置
        chart_config = viz_config.get("charts", {})
        self.colors = chart_config.get("default_colors", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
        ])

        # 创建Dash应用
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="微博舆情风险监控系统"
        )

        # 数据存储
        self.data: Optional[pd.DataFrame] = None
        self.risk_data: Optional[Dict] = None

        # 设置布局
        self._setup_layout()

        # 设置回调
        self._setup_callbacks()

        logger.info("可视化仪表盘初始化完成")

    def _setup_layout(self):
        """设置仪表盘布局"""
        self.app.layout = dbc.Container([
            # 标题行
            dbc.Row([
                dbc.Col([
                    html.H1("微博舆情风险监控系统", className="text-center my-4"),
                    html.P("实时监控社会风险感知与舆情变化", className="text-center text-muted")
                ])
            ]),

            # 指标卡片行
            dbc.Row([
                dbc.Col([
                    self._create_metric_card("total-posts", "总微博数", "0", "primary")
                ], md=3),
                dbc.Col([
                    self._create_metric_card("risk-score", "风险指数", "0", "danger")
                ], md=3),
                dbc.Col([
                    self._create_metric_card("sentiment-score", "情感指数", "0", "success")
                ], md=3),
                dbc.Col([
                    self._create_metric_card("active-users", "活跃用户", "0", "info")
                ], md=3)
            ], className="mb-4"),

            # 图表行 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("情感趋势"),
                        dbc.CardBody([
                            dcc.Graph(id="sentiment-trend-chart")
                        ])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("情感分布"),
                        dbc.CardBody([
                            dcc.Graph(id="sentiment-distribution-chart")
                        ])
                    ])
                ], md=4)
            ], className="mb-4"),

            # 图表行 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("风险维度分析"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-dimension-chart")
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("情绪分布"),
                        dbc.CardBody([
                            dcc.Graph(id="emotion-distribution-chart")
                        ])
                    ])
                ], md=6)
            ], className="mb-4"),

            # 图表行 3
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("热门话题"),
                        dbc.CardBody([
                            dcc.Graph(id="topic-chart")
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("传播热度"),
                        dbc.CardBody([
                            dcc.Graph(id="engagement-chart")
                        ])
                    ])
                ], md=6)
            ], className="mb-4"),

            # 预警信息
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("风险预警", className="bg-warning text-white"),
                        dbc.CardBody([
                            html.Div(id="warning-messages")
                        ])
                    ])
                ])
            ], className="mb-4"),

            # 数据表格
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("最新微博"),
                        dbc.CardBody([
                            html.Div(id="latest-posts-table")
                        ])
                    ])
                ])
            ]),

            # 定时更新
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            ),

            # 数��存储
            dcc.Store(id='data-store')

        ], fluid=True, className="py-3")

    def _create_metric_card(
        self,
        card_id: str,
        title: str,
        value: str,
        color: str = "primary"
    ) -> dbc.Card:
        """创建指标卡片"""
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="card-subtitle mb-2 text-muted"),
                html.H3(value, id=card_id, className=f"card-title text-{color}")
            ])
        ], className="text-center")

    def _setup_callbacks(self):
        """设置回调函数"""

        @self.app.callback(
            [Output("total-posts", "children"),
             Output("risk-score", "children"),
             Output("sentiment-score", "children"),
             Output("active-users", "children"),
             Output("sentiment-trend-chart", "figure"),
             Output("sentiment-distribution-chart", "figure"),
             Output("risk-dimension-chart", "figure"),
             Output("emotion-distribution-chart", "figure"),
             Output("topic-chart", "figure"),
             Output("engagement-chart", "figure"),
             Output("warning-messages", "children"),
             Output("latest-posts-table", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_dashboard(n):
            """更新仪表盘"""
            # 使用示例数据（实际应用中从数据库加载）
            if self.data is None:
                df = self._generate_sample_data()
            else:
                df = self.data

            # 计算指标
            total_posts = len(df)
            risk_score = self._calculate_risk_score(df)
            sentiment_score = df["sentiment_score"].mean() if "sentiment_score" in df.columns else 0.5
            active_users = df["user_id"].nunique() if "user_id" in df.columns else 0

            # 生成图表
            sentiment_trend = self._create_sentiment_trend_chart(df)
            sentiment_dist = self._create_sentiment_distribution_chart(df)
            risk_radar = self._create_risk_radar_chart()
            emotion_dist = self._create_emotion_distribution_chart(df)
            topic_chart = self._create_topic_chart(df)
            engagement_chart = self._create_engagement_chart(df)

            # 预警信息
            warnings = self._generate_warnings(risk_score, df)

            # 最新微博表格
            posts_table = self._create_posts_table(df)

            return (
                f"{total_posts:,}",
                f"{risk_score:.1f}",
                f"{sentiment_score:.2f}",
                f"{active_users:,}",
                sentiment_trend,
                sentiment_dist,
                risk_radar,
                emotion_dist,
                topic_chart,
                engagement_chart,
                warnings,
                posts_table
            )

    def _generate_sample_data(self, n: int = 200) -> pd.DataFrame:
        """生成示例数据"""
        np.random.seed(42)

        dates = pd.date_range(end=datetime.now(), periods=n, freq="H")

        data = {
            "weibo_id": [f"wb_{i}" for i in range(n)],
            "user_id": [f"user_{np.random.randint(1, 50)}" for _ in range(n)],
            "user_name": [f"用户{np.random.randint(1, 50)}" for _ in range(n)],
            "content": [f"这是第{i}条微博内容" for i in range(n)],
            "created_at": dates,
            "sentiment_score": np.random.uniform(-1, 1, n),
            "sentiment_polarity": np.random.choice(["positive", "neutral", "negative"], n),
            "emotion": np.random.choice(["joy", "anger", "fear", "sadness", "neutral"], n),
            "reposts_count": np.random.randint(0, 1000, n),
            "comments_count": np.random.randint(0, 500, n),
            "attitudes_count": np.random.randint(0, 5000, n)
        }

        return pd.DataFrame(data)

    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """计算风险分数"""
        if df.empty:
            return 0.0

        # 基于负面情感比例
        negative_ratio = (df.get("sentiment_polarity", pd.Series()) == "negative").mean()

        # 基于情绪
        fear_ratio = (df.get("emotion", pd.Series()) == "fear").mean()
        anger_ratio = (df.get("emotion", pd.Series()) == "anger").mean()

        score = (negative_ratio * 40 + fear_ratio * 30 + anger_ratio * 30)

        return min(100, score * 100)

    def _create_sentiment_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建情感趋势图"""
        if df.empty or "created_at" not in df.columns:
            return go.Figure()

        df = df.copy()
        df["created_at"] = pd.to_datetime(df["created_at"])

        # 按小时聚合
        hourly = df.set_index("created_at").resample("H").agg({
            "sentiment_score": "mean",
            "weibo_id": "count"
        }).reset_index()
        hourly.columns = ["时间", "情感得分", "微博数量"]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=hourly["时间"],
                y=hourly["情感得分"],
                name="情感得分",
                line=dict(color=self.colors[0])
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                x=hourly["时间"],
                y=hourly["微博数量"],
                name="微博数量",
                opacity=0.3,
                marker_color=self.colors[1]
            ),
            secondary_y=True
        )

        fig.update_layout(
            xaxis_title="时间",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        fig.update_yaxes(title_text="情感得分", secondary_y=False)
        fig.update_yaxes(title_text="微博数量", secondary_y=True)

        return fig

    def _create_sentiment_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建情感分布图"""
        if "sentiment_polarity" not in df.columns:
            return go.Figure()

        dist = df["sentiment_polarity"].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=["正面", "中性", "负面"],
            values=[
                dist.get("positive", 0),
                dist.get("neutral", 0),
                dist.get("negative", 0)
            ],
            hole=0.4,
            marker_colors=[self.colors[2], self.colors[4], self.colors[3]]
        )])

        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))

        return fig

    def _create_risk_radar_chart(self) -> go.Figure:
        """创建风险雷达图"""
        categories = ["健康风险", "经济风险", "社会风险", "政治风险", "环境风险"]

        # 示例数据
        values = [45, 60, 35, 25, 30]
        values.append(values[0])  # 闭合
        categories.append(categories[0])

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color=self.colors[0]
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def _create_emotion_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建情绪分布图"""
        if "emotion" not in df.columns:
            return go.Figure()

        dist = df["emotion"].value_counts()

        emotion_labels = {
            "joy": "喜悦",
            "anger": "愤怒",
            "fear": "恐惧",
            "sadness": "悲伤",
            "surprise": "惊讶",
            "disgust": "厌恶",
            "neutral": "中性"
        }

        fig = go.Figure(data=[go.Bar(
            x=[emotion_labels.get(e, e) for e in dist.index],
            y=dist.values,
            marker_color=self.colors
        )])

        fig.update_layout(
            xaxis_title="情绪类型",
            yaxis_title="数量",
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def _create_topic_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建话题图表"""
        # 示例热门话题
        topics = ["疫情防控", "经济发展", "教育改革", "环境保护", "社会民生"]
        counts = [150, 120, 80, 60, 40]

        fig = go.Figure(data=[go.Bar(
            y=topics[::-1],
            x=counts[::-1],
            orientation='h',
            marker_color=self.colors[0]
        )])

        fig.update_layout(
            xaxis_title="提及次数",
            margin=dict(l=100, r=40, t=40, b=40)
        )

        return fig

    def _create_engagement_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建互动热度图"""
        if df.empty:
            return go.Figure()

        df = df.copy()
        df["总互动"] = (
            df.get("reposts_count", 0) +
            df.get("comments_count", 0) +
            df.get("attitudes_count", 0)
        )

        # 按时间聚合
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            hourly = df.set_index("created_at").resample("H")["总互动"].sum().reset_index()

            fig = go.Figure(data=[go.Scatter(
                x=hourly["created_at"],
                y=hourly["总互动"],
                mode='lines+markers',
                fill='tozeroy',
                line_color=self.colors[1]
            )])

            fig.update_layout(
                xaxis_title="时间",
                yaxis_title="互动量",
                margin=dict(l=40, r=40, t=40, b=40)
            )

            return fig

        return go.Figure()

    def _generate_warnings(self, risk_score: float, df: pd.DataFrame) -> List:
        """生成预警信息"""
        warnings = []

        if risk_score >= 70:
            warnings.append(
                dbc.Alert(
                    f"⚠️ 高风险预警：当前风险指数为 {risk_score:.1f}，请立即关注！",
                    color="danger"
                )
            )
        elif risk_score >= 50:
            warnings.append(
                dbc.Alert(
                    f"⚠️ 中风险提示：当前风险指数为 {risk_score:.1f}，建议密切监控",
                    color="warning"
                )
            )

        # 检查负面情绪
        if "emotion" in df.columns:
            fear_count = (df["emotion"] == "fear").sum()
            anger_count = (df["emotion"] == "anger").sum()

            if fear_count > len(df) * 0.2:
                warnings.append(
                    dbc.Alert(f"⚠️ 恐惧情绪异常：检测到 {fear_count} 条恐惧相关内容", color="warning")
                )
            if anger_count > len(df) * 0.2:
                warnings.append(
                    dbc.Alert(f"⚠️ 愤怒情绪异常：检测到 {anger_count} 条愤怒相关内容", color="warning")
                )

        if not warnings:
            warnings.append(
                dbc.Alert("✅ 当前舆情态势平稳，未检测到异常", color="success")
            )

        return warnings

    def _create_posts_table(self, df: pd.DataFrame) -> dbc.Table:
        """创建微博表格"""
        if df.empty:
            return html.P("暂无数据")

        # 取最新10条
        recent = df.sort_values("created_at", ascending=False).head(10)

        return dbc.Table([
            html.Thead(html.Tr([
                html.Th("时间"),
                html.Th("用户"),
                html.Th("内容"),
                html.Th("情感"),
                html.Th("互动")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(str(row.get("created_at", ""))[:16]),
                    html.Td(row.get("user_name", "")),
                    html.Td(str(row.get("content", ""))[:50] + "..."),
                    html.Td(row.get("sentiment_polarity", "")),
                    html.Td(row.get("reposts_count", 0) + row.get("comments_count", 0))
                ])
                for _, row in recent.iterrows()
            ])
        ], bordered=True, hover=True, striped=True, size="sm")

    def update_data(self, df: pd.DataFrame):
        """更新数据"""
        self.data = df
        logger.info(f"仪表盘数据已更新: {len(df)} 条")

    def update_risk_data(self, risk_data: Dict):
        """更新风险数据"""
        self.risk_data = risk_data

    def run(self, host: str = None, port: int = None, debug: bool = None):
        """启动仪表盘"""
        if host is None:
            host = self.host
        if port is None:
            port = self.port
        if debug is None:
            debug = self.debug

        logger.info(f"启动仪表盘: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run(debug=True)

