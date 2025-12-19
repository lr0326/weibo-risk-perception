"""
情绪趋势预测模型
基于LSTM的时间序列预测
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("weibo_risk")

from src.utils.helpers import load_config


@dataclass
class PredictionResult:
    """预测结果"""
    timestamps: List
    predictions: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    metrics: Dict


class LSTMModel(nn.Module):
    """LSTM预测模型"""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])

        return out


class EmotionTrendPredictor:
    """
    情绪趋势预测器

    功能：
    - 时间序列预处理
    - LSTM模型训练
    - 趋势预测
    - 异常检测
    """

    def __init__(
        self,
        method: str = "lstm",
        config_path: str = "config/config.yaml"
    ):
        """
        初始化预测器

        Args:
            method: 预测方法 (lstm, arima, prophet)
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.method = method.lower()

        # LSTM配置
        lstm_config = self.config.get("models", {}).get("lstm", {})
        self.hidden_dim = lstm_config.get("hidden_dim", 128)
        self.num_layers = lstm_config.get("num_layers", 2)
        self.dropout = lstm_config.get("dropout", 0.2)
        self.sequence_length = lstm_config.get("sequence_length", 24)
        self.learning_rate = lstm_config.get("learning_rate", 0.001)
        self.epochs = lstm_config.get("epochs", 100)
        self.early_stopping_patience = lstm_config.get("early_stopping_patience", 10)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型
        self.model: Optional[LSTMModel] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # 模型路径
        self.model_path = self.config.get("paths", {}).get("data", {}).get("models", "data/models")

        logger.info(f"情绪趋势预测器初始化完成，使用方法: {self.method}")

    def prepare_time_series(
        self,
        df: pd.DataFrame,
        value_column: str = "sentiment_score",
        time_column: str = "created_at",
        freq: str = "H"
    ) -> pd.DataFrame:
        """
        准备时间序列数据

        Args:
            df: 输入数据
            value_column: 值列
            time_column: 时间列
            freq: 聚合频率 (H: 小时, D: 天, W: 周)

        Returns:
            时间序列DataFrame
        """
        df = df.copy()

        # 转换时间
        df[time_column] = pd.to_datetime(df[time_column])

        # 设置索引
        df = df.set_index(time_column)

        # 按频率聚合
        ts = df[[value_column]].resample(freq).agg({
            value_column: 'mean'
        })

        # 填充缺失值
        ts = ts.interpolate(method='linear')
        ts = ts.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"时间序列准备完成: {len(ts)} 个时间点")

        return ts

    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据

        Args:
            data: 输入数据
            sequence_length: 序列长度

        Returns:
            X, y 序列数据
        """
        if sequence_length is None:
            sequence_length = self.sequence_length

        X, y = [], []

        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])

        return np.array(X), np.array(y)

    def train_lstm(
        self,
        time_series: pd.DataFrame,
        sequence_length: int = None,
        epochs: int = None,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """
        训练LSTM模型

        Args:
            time_series: 时间序列数据
            sequence_length: 序列长度
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例

        Returns:
            训练结果
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        if epochs is None:
            epochs = self.epochs

        # 准备数据
        values = time_series.values
        scaled_values = self.scaler.fit_transform(values)

        X, y = self.create_sequences(scaled_values, sequence_length)

        # 划分训练集和验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        input_dim = X.shape[2]
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1,
            dropout=self.dropout
        ).to(self.device)

        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 训练
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)

            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                logger.info(f"早停于第 {epoch + 1} 轮")
                break

            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch [{epoch + 1}/{epochs}], "
                           f"Train Loss: {avg_train_loss:.6f}, "
                           f"Val Loss: {val_loss:.6f}")

        # 计算最终指标
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train).cpu().numpy()
            val_pred = self.model(X_val).cpu().numpy()

        # 逆变换
        train_pred_inv = self.scaler.inverse_transform(train_pred)
        val_pred_inv = self.scaler.inverse_transform(val_pred)
        y_train_inv = self.scaler.inverse_transform(y_train.cpu().numpy())
        y_val_inv = self.scaler.inverse_transform(y_val.cpu().numpy())

        # 计算RMSE
        train_rmse = np.sqrt(np.mean((train_pred_inv - y_train_inv) ** 2))
        val_rmse = np.sqrt(np.mean((val_pred_inv - y_val_inv) ** 2))

        result = {
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "epochs_trained": len(train_losses)
        }

        logger.info(f"LSTM训练完成 - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

        return result

    def predict_future(
        self,
        time_series: pd.DataFrame,
        steps: int = 24,
        return_confidence: bool = True
    ) -> PredictionResult:
        """
        预测未来趋势

        Args:
            time_series: 历史时间序列
            steps: 预测步数
            return_confidence: 是否返回置信区间

        Returns:
            PredictionResult对象
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train_lstm方法")

        self.model.eval()

        # 准备最后的序列
        values = time_series.values
        scaled_values = self.scaler.transform(values)

        current_seq = scaled_values[-self.sequence_length:]
        current_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)

        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_seq)
                predictions.append(pred.cpu().numpy()[0, 0])

                # 更新序列
                new_seq = torch.cat([
                    current_seq[:, 1:, :],
                    pred.unsqueeze(1)
                ], dim=1)
                current_seq = new_seq

        # 逆变换
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_inv = self.scaler.inverse_transform(predictions).flatten()

        # 生成时间戳
        last_time = time_series.index[-1]
        freq = pd.infer_freq(time_series.index)
        if freq is None:
            freq = 'H'
        future_times = pd.date_range(start=last_time, periods=steps + 1, freq=freq)[1:]

        # 置信区间（简化估计）
        if return_confidence:
            std = np.std(time_series.values)
            confidence_lower = (predictions_inv - 1.96 * std).tolist()
            confidence_upper = (predictions_inv + 1.96 * std).tolist()
        else:
            confidence_lower = confidence_upper = []

        return PredictionResult(
            timestamps=future_times.tolist(),
            predictions=predictions_inv.tolist(),
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            metrics={"steps": steps}
        )

    def detect_anomalies(
        self,
        time_series: pd.DataFrame,
        threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        检测异常点

        Args:
            time_series: 时间序列
            threshold: 异常阈值（标准差倍数）

        Returns:
            包含异常标记的DataFrame
        """
        ts = time_series.copy()
        values = ts.values.flatten()

        # 计算统计量
        mean = np.mean(values)
        std = np.std(values)

        # 标记异常
        ts['z_score'] = (values - mean) / std
        ts['is_anomaly'] = np.abs(ts['z_score']) > threshold

        anomaly_count = ts['is_anomaly'].sum()
        logger.info(f"检测到 {anomaly_count} 个异常点 (阈值: {threshold}σ)")

        return ts

    def save_model(self, filename: str = "lstm_model.pt"):
        """保存模型"""
        if self.model is None:
            logger.warning("模型未训练，无法保存")
            return

        filepath = os.path.join(self.model_path, filename)
        os.makedirs(self.model_path, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            }
        }, filepath)

        logger.info(f"模型已保存至 {filepath}")

    def load_model(self, filename: str = "lstm_model.pt"):
        """加载模型"""
        filepath = os.path.join(self.model_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        config = checkpoint['config']
        self.model = LSTMModel(
            input_dim=1,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.sequence_length = config['sequence_length']

        logger.info(f"模型已从 {filepath} 加载")


if __name__ == "__main__":
    # 测试
    predictor = EmotionTrendPredictor(method="lstm")

    # 生成模拟时间序列
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="H")
    values = np.sin(np.arange(200) * 0.1) * 0.3 + 0.5 + np.random.randn(200) * 0.05

    ts = pd.DataFrame({"sentiment_score": values}, index=dates)

    print(f"时间序列形状: {ts.shape}")

    # 训练
    result = predictor.train_lstm(ts, sequence_length=24, epochs=30)
    print(f"训练结果: {result}")

    # 预测
    predictions = predictor.predict_future(ts, steps=12)
    print(f"\n预测未来12小时:")
    for t, p in zip(predictions.timestamps[:5], predictions.predictions[:5]):
        print(f"  {t}: {p:.4f}")

