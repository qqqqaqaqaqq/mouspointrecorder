import torch

from collections import deque
from datetime import datetime

from app.models.CnnDenseLstm import CnnDenseLstm, CnnDenseLstmOneClass

from app.ui.train import points_to_features_scaled

from app.services.indicators import indicators_generation

import pandas as pd
import app.core.globals as globals

class MacroDetector:
    def __init__(self, model_path: str, seq_len=globals.SEQ_LEN, threshold=0.8, device=None):
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ===== 모델 초기화 =====
        # labeling
        self.model = CnnDenseLstm(
            input_size=len(globals.FEACTURE), 
            lstm_hidden_size=globals.lstm_hidden_size, 
            lstm_layers=globals.lstm_layers, 
            dropout=globals.dropout
        )

        # oneclass
        # self.model = CnnDenseLstmOneClass(
        #     input_size=len(globals.FEACTURE), 
        #     lstm_hidden_size=globals.lstm_hidden_size, 
        #     lstm_layers=globals.lstm_layers, 
        #     dropout=globals.dropout
        # )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # ===== 좌표 buffer =====
        self.buffer = deque(maxlen=seq_len * 3)
        self.prev_speed = 0.0


    def compress_zero_sum_rows(self, df: pd.DataFrame, max_zeros=5):
        df_compressed = []
        zero_count = 0

        for _, row in df.iterrows():
            if row.sum() == 0:  # 행 전체 합이 0이면
                zero_count += 1
                if zero_count <= max_zeros:
                    df_compressed.append(row)
            else:
                zero_count = 0
                df_compressed.append(row)

        return pd.DataFrame(df_compressed).reset_index(drop=True)

    def push(self, data:dict):
        self.buffer.append((data.get('x'), data.get('y'), data.get('timestamp'), data.get('event_type'), data.get('is_pressed')))

        if len(self.buffer) < self.seq_len * 3:
            return None
        
        return self._infer()

    def _infer(self):
        xs = [p[0] for p in self.buffer]
        ys = [p[1] for p in self.buffer]
        ts = [p[2] for p in self.buffer] 
        event_type = [p[3] for p in self.buffer] 
        is_pressed = [p[4] for p in self.buffer]

        df:pd.DataFrame = pd.DataFrame({"timestamp": ts, "x": xs, "y": ys, "event_type" : event_type, "is_pressed":is_pressed })
        df = indicators_generation(df)

        df = df.sort_values('timestamp').reset_index(drop=True)

        df = df[globals.FEACTURE].copy()

 
        # ===== 0 압축 =====
        df = self.compress_zero_sum_rows(df, max_zeros=1)
        
        # ===== Feature 필터 =====
        
        X_infer, _, = points_to_features_scaled(df_chunk=df, seq_len=globals.SEQ_LEN, stride=globals.STRIDE)

        if X_infer.size == 0:
            return None

        # 마지막 시퀀스만 사용
        X_tensor = torch.tensor(X_infer[-1], dtype=torch.float32).unsqueeze(0).to(self.device)


        # labeling
        with torch.no_grad():
            pred = self.model(X_tensor)
            prob = torch.sigmoid(pred).squeeze().item()  # 0~1

        # oneclass
        # with torch.no_grad():
        #     pred = self.model(X_tensor)  # (1, seq_len, feature)
        #     recon_error = torch.mean((pred - X_tensor)**2)  # 스칼라
        #     prob = torch.sigmoid(-recon_error)  # 값이 0~1, 작을수록 정상

        return {"is_macro": prob < self.threshold, "prob": prob}

