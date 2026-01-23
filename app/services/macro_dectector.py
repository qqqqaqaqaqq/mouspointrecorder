import torch
import numpy as np
from collections import deque
from datetime import datetime

from app.models.DenseLstm import DenseLSTMModel

class MacroDetector:
    def __init__(
        self,
        model_path: str,
        seq_len=15,
        threshold=0.8,
        device=None
    ):
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        self.model = DenseLSTMModel(
            input_size=6,
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

        # 최근 좌표 버퍼
        self.buffer = deque(maxlen=seq_len + 1)
        self.prev_speed = 0.0

    def push(self, x: int, y: int, timestamp: datetime):
        """좌표 하나 넣고, 판단 가능하면 결과 반환"""
        self.buffer.append((x, y, timestamp))

        if len(self.buffer) < self.seq_len + 1:
            return None

        return self._infer()

    def _infer(self):
        features = []

        xs = [p[0] for p in self.buffer]
        ys = [p[1] for p in self.buffer]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        prev_x, prev_y, prev_t = self.buffer[0]
        prev_speed = self.prev_speed

        for x, y, t in list(self.buffer)[1:]:
            dt = (t - prev_t).total_seconds()
            if dt <= 0:
                dt = 1e-6

            dx = (x - prev_x) / (x_max - x_min + 1e-6)
            dy = (y - prev_y) / (y_max - y_min + 1e-6)

            speed = np.sqrt(dx**2 + dy**2) / dt
            acc = speed - prev_speed
            angle = np.arctan2(dy, dx)

            features.append([dt, dx, dy, speed, acc, angle])

            prev_x, prev_y, prev_t = x, y, t
            prev_speed = speed

        self.prev_speed = prev_speed

        x_tensor = torch.tensor(
            np.array(features, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(x_tensor)
            prob = torch.sigmoid(logit).item()

        return {
            "prob": prob,
            "is_macro": prob >= self.threshold
        }
