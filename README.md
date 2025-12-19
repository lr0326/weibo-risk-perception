# 基于微博数据的社会风险感知与舆情预测系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 项目简介

本项目构建了一个基于微博数据的多维度社会风险感知模型，通过大数据分析和机器学习技术，实现对公众情绪变化趋势的动态预测和舆情预警。

### 核心功能

- 🔍 **多源数据采集**：微博API、网络爬虫、实时流数据采集
- 🧹 **智能数据处理**：文本清洗、分词、特征工程
- 🎯 **情感分析**：基于BERT的多维度情感识别
- 📊 **风险感知建模**：多维度风险评估模型
- 🔮 **趋势预测**：LSTM时间序列预测
- ⚠️ **预警系统**：实时风险评估与预警
- 📈 **可视化仪表盘**：交互式数据展示

### 应用场景

- 公共卫生应急管理
- 舆情监测与预警
- 社会治理决策支持
- 传播学研究
- 健康传播策略优化

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      数据采集层                              │
│  微博API │ 网络爬虫 │ 实时流数据 │ 历史数据归档              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      数据处理层                              │
│  清洗 │ 去重 │ 分词 │ 特征提取 │ 情感标注                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      模型分析层                              │
│  风险感知模型 │ 情绪预测模型 │ 传播动力学模型              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      可视化展示层                            │
│  实时监控仪表盘 │ 预警系统 │ 报告生成                      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
weibo-risk-perception/
├── README.md                      # 项目说明文档
├── requirements.txt               # Python依赖包
├── setup.py                       # 安装配置
├── . gitignore                     # Git忽略文件
├── config/                        # 配置文件目录
│   ├── config.yaml               # 主配置文件
│   ├── api_config.yaml           # API配置
│   └── model_config. yaml         # 模型配置
├── data/                          # 数据目录
│   ├── raw/                      # 原始数据
│   ├── processed/                # 处理后数据
│   ├── models/                   # 训练好的模型
│   └── outputs/                  # 输出结果
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── data_collection/          # 数据采集模块
│   │   ├── __init__.py
│   │   ├── weibo_collector.py   # 微博数据采集器
│   │   └── stream_collector.py  # 实时流采集
│   ├── preprocessing/            # 数据预处理模块
│   │   ├── __init__.py
│   │   ├── text_cleaner.py      # 文本清洗
│   │   └── feature_extractor.py # 特征提取
│   ├── analysis/                 # 分析模块
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py    # 情感分析
│   │   ├── risk_perception. py       # 风险感知分析
│   │   └── network_analysis.py      # 社会网络分析
│   ├── models/                   # 模型模块
│   │   ├── __init__.py
│   │   ├── risk_model.py        # 风险感知模型
│   │   ├── prediction_model.py  # 预测模型
│   │   └── clustering. py        # 聚类模型
│   ├── visualization/            # 可视化模块
│   │   ├── __init__.py
│   │   ├── dashboard.py         # 仪表盘
│   │   └── report_generator.py  # 报告生成
│   ├── utils/                    # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py            # 日志工具
│   │   ├── database.py          # 数据库操作
│   │   └── helpers.py           # 辅助函数
│   └── pipeline. py               # 主流程
├── notebooks/                     # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_result_analysis.ipynb
├── tests/                         # 测试代码
│   ├── __init__.py
│   ├── test_collector.py
│   ├── test_preprocessor.py
│   └── test_models.py
├── scripts/                       # 脚本文件
│   ├── run_collection.py        # 数据采集脚本
│   ├── run_analysis.py          # 分析脚本
│   └── run_dashboard.py         # 启动仪表盘
├── docs/                          # 文档目录
│   ├── installation.md          # 安装指南
│   ├── user_guide.md            # 使用指南
│   ├── api_reference.md         # API文档
│   └── methodology.md           # 方法论说明
└── examples/                      # 示例代码
    ├── basic_usage.py
    ├── custom_analysis.py
    └── batch_processing.py
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 或 conda
- 4GB+ RAM
- (可选) CUDA支持的GPU

### 安装步骤

#### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/weibo-risk-perception. git
cd weibo-risk-perception
```

#### 2. 创建虚拟环境

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 使用conda
conda create -n weibo-risk python=3.8
conda activate weibo-risk
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt

# 如果需要GPU支持
pip install -r requirements-gpu.txt
```

#### 4. 配置API密钥

复制配置模板并填入您的API信息：

```bash
cp config/api_config. yaml.template config/api_config.yaml
```

编辑 `config/api_config.yaml`：

```yaml
weibo: 
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_API_SECRET"
  access_token: "YOUR_ACCESS_TOKEN"
```

#### 5. 下载预训练模型

```bash
python scripts/download_models.py
```

### 基础使用

#### 数据采集

```python
from src.data_collection.weibo_collector import WeiboDataCollector

# 初始化采集器
collector = WeiboDataCollector(access_token="YOUR_TOKEN")

# 采集数据
data = collector. search_weibo_by_keyword(
    keyword="新冠疫情",
    count=100,
    pages=10
)

# 保存数据
data.to_csv("data/raw/weibo_data. csv", index=False)
```

#### 情感分析

```python
from src.analysis.sentiment_analyzer import SentimentAnalyzer

# 初始化分析器
analyzer = SentimentAnalyzer(model_type='bert')

# 分析单条文本
text = "今天的疫情防控措施很到位，大家都很配合"
result = analyzer.analyze_sentiment(text)

print(f"情感极性: {result['polarity']}")
print(f"主要情绪: {result['emotion']}")
print(f"情感强度:  {result['intensity']}")
```

#### 风险感知建模

```python
from src.models.risk_model import MultiDimensionalRiskModel
import pandas as pd

# 加载数据
df = pd.read_csv("data/processed/features.csv")

# 初始化模型
model = MultiDimensionalRiskModel()

# 构建特征矩阵
features = model.build_feature_matrix(df)

# 群体细分
clusters, profiles = model.segment_population(features, n_clusters=5)

# 查看群体画像
for label, profile in profiles.items():
    print(f"\n{label}:")
    print(f"  规模: {profile['size']}")
    print(f"  风险感知:  {profile['avg_risk_perception']:.2f}")
    print(f"  特征: {profile['characteristics']}")
```

#### 趋势预测

```python
from src.models.prediction_model import EmotionTrendPredictor

# 初始化预测器
predictor = EmotionTrendPredictor(method='lstm')

# 准备时间序列
time_series = predictor.prepare_time_series(df, freq='H')

# 训练模型
predictor.train_lstm(time_series, sequence_length=24, epochs=50)

# 预测未来24小时
predictions = predictor.predict_future(time_series, steps=24)

print(predictions)
```

#### 启动可视化仪表盘

```bash
python scripts/run_dashboard.py --port 8050
```

然后在浏览器访问 `http://localhost:8050`

## 📊 完整流程示例

```python
from src.pipeline import RiskPerceptionPipeline
from datetime import datetime, timedelta

# 初始化流水线
pipeline = RiskPerceptionPipeline()

# 设置分析参数
keywords = "新冠疫情"
start_date = datetime. now() - timedelta(days=7)
end_date = datetime.now()

# 运行完整分析
results = pipeline.run_analysis(
    keywords=keywords,
    start_date=start_date,
    end_date=end_date
)

# 查看结果
print(f"风险等级: {results['risk_level']}")
print(f"风险得分: {results['risk_score']}")
print(f"预警信息: {results['warnings']}")
print(f"应对建议: {results['recommendations']}")

# 生成报告
pipeline.dashboard. generate_report(
    results,
    output_path='data/outputs/report.html'
)
```

## 🔧 配置说明

### 主配置文件 (config/config.yaml)

```yaml
# 数据采集配置
data_collection:
  batch_size: 100
  max_pages: 50
  retry_times: 3
  sleep_interval: 1

# 模型配置
models:
  sentiment: 
    model_name: "bert-base-chinese"
    max_length: 512
    batch_size: 32
  
  lstm:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2
    sequence_length: 24
  
  clustering:
    n_clusters: 5
    algorithm: "kmeans"

# 预警阈值
warning_thresholds:
  risk_perception:  0.7
  fear_level: 0.6
  anger_level: 0.5
  volume_spike: 3. 0

# 可视化配置
visualization: 
  update_interval: 300  # 秒
  port: 8050
  debug: false
```

## 📈 数据格式

### 输入数据格式

微博数据应包含以下字段：

```python
{
    'weibo_id': str,          # 微博ID
    'user_id': str,           # 用户ID
    'user_name': str,         # 用户名
    'user_followers': int,    # 粉丝数
    'user_verified': bool,    # 是否认证
    'content': str,           # 微博内容
    'created_at': datetime,   # 发布时间
    'reposts_count': int,     # 转发数
    'comments_count': int,    # 评论数
    'attitudes_count': int,   # 点赞数
    'location': str,          # 地理位置
    'source': str,            # 来源
    'pic_urls': list,         # 图片URL列表
    'is_repost': bool         # 是否为转发
}
```

### 输出结果格式

```python
{
    'analysis_date': datetime,
    'sample_size': int,
    'risk_level': str,        # 'low', 'medium', 'high', 'critical'
    'risk_score': float,      # 0-100
    'sentiment_summary': {
        'avg_polarity': float,
        'dominant_emotion': str,
        'emotion_distribution': dict
    },
    'cluster_profiles': dict,
    'predictions': DataFrame,
    'warnings': list,
    'recommendations': list
}
```

## 🧪 测试

运行所有测试：

```bash
pytest tests/
```

运行特定测试：

```bash
pytest tests/test_collector.py -v
```

生成覆盖率报告：

```bash
pytest --cov=src tests/
```

## 📚 详细文档

- [安装指南](docs/installation.md)
- [使用教程](docs/user_guide.md)
- [API参考](docs/api_reference.md)
- [方法论说明](docs/methodology.md)
- [常见问题](docs/faq.md)

## 🤝 贡献指南

我们欢迎任何形式的贡献！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

请确保：
- 代码遵循 PEP 8 规范
- 添加必要的测试
- 更新相关文档

## 📝 更新日志

### v1.0.0 (2025-12-18)

- ✨ 初始版本发布
- 🔍 实现微博数据采集功能
- 🎯 集成BERT情感分析模型
- 📊 构建多维度风险感知模型
- 🔮 实现LSTM趋势预测
- ⚠️ 添加预警系统
- 📈 创建可视化仪表盘

## 🔒 隐私与伦理

本项目在数据采集和分析过程中严格遵守：

- ✅ 仅采集公开数据
- ✅ 遵守微博API使用条款
- ✅ 保护用户隐私，不公开个人信息
- ✅ 数据仅用于学术研究和公益目的
- ✅ 遵守《个人信息保护法》等相关法律法规

## ⚠️ 免责声明

- 本项目仅供学术研究和教育目的使用
- 分析结果仅供参考，不构成任何决策依据
- 使用者需自行承担使用本系统产生的风险
- 请遵守当地法律法规和平台使用协议

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 👥 作者

- **项目负责人** - lr0326
- **贡献者列表** - [Contributors](https://github.com/yourusername/weibo-risk-perception/contributors)

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - BERT模型支持
- [Plotly](https://plotly.com/) - 可视化框架
- [Jieba](https://github.com/fxsjy/jieba) - 中文分词
- 所有贡献者和支持者

## 📮 联系方式

- 项目主页: https://github.com/yourusername/weibo-risk-perception
- 问题反馈: [Issues](https://github.com/yourusername/weibo-risk-perception/issues)
- 邮箱: your.email@example.com

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/weibo-risk-perception&type=Date)](https://star-history.com/#yourusername/weibo-risk-perception&Date)

---

如果这个项目对您有帮助，请给我们一个 ⭐️ Star！