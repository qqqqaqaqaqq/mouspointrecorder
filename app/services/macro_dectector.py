import torch
import numpy as np
from collections import deque
from datetime import datetime
import joblib

from sklearn.preprocessing import StandardScaler
from app.models.DenseLstm import DenseLSTMModel
from app.models.CnnDenseLstm import CnnDenseLstm
from sklearn.preprocessing import StandardScaler

from app.ui.train import points_to_features_scaled

from app.services.indicators import indicators_generation

import pandas as pd
import app.core.globals as globals
import app.ui.train as train

class MacroDetector:
    def __init__(self, model_path: str, seq_len=globals.SEQ_LEN, threshold=0.8, device=None):
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ===== 모델 초기화 =====
        self.model = CnnDenseLstm(
            input_size=len(globals.FEACTURE), 
            lstm_hidden_size=128, 
            lstm_layers=3, 
            dropout=0.3
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # ===== 좌표 buffer =====
        self.buffer = deque(maxlen=seq_len * 3)
        self.prev_speed = 0.0


    def push(self, x: int, y: int, timestamp: datetime):
        self.buffer.append((x, y, timestamp))

        if len(self.buffer) < self.seq_len * 3:
            return None
        
        return self._infer()

    def _infer(self):
        xs = [p[0] for p in self.buffer]
        ys = [p[1] for p in self.buffer]
        ts = [p[2] for p in self.buffer]  # timestamp 그대로

        df:pd.DataFrame = pd.DataFrame({"timestamp": ts, "x": xs, "y": ys})
        df = indicators_generation(df)

        df = df.sort_values('timestamp').reset_index(drop=True)

        df = df[globals.FEACTURE].copy()
        
        # ===== Feature 필터 =====
        
        X_infer, _ = points_to_features_scaled(train_df=df, seq_len=globals.SEQ_LEN, stride=globals.STRIDE)

        
        if X_infer.size == 0:  # 배열 비었으면 None 반환
            return None

        # 마지막 시퀀스만 사용
        X_tensor = torch.tensor(X_infer[-1], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(X_tensor)
            prob = torch.sigmoid(pred).squeeze().item()  # 0~1

        return {"is_macro": prob < self.threshold, "prob": prob}

