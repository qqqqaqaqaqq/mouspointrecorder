import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import traceback
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import app.core.globals as globals

from sklearn.preprocessing import StandardScaler

# 모델
from app.models.MousePoint import MousePoint, MacroMousePoint
from app.models.DenseLstm import DenseLSTMModel
from app.models.CnnDenseLstm import CnnDenseLstm
from app.models.MouseFeatureDataset import MouseFeatureDataset

import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController

from app.services.indicators import indicators_generation

save_path = "app/models/weights/mouse_macro_lstm_best.pt"

# stride = > seq 데이터 겹치는 양 (How much each sequence shifts forward)
def points_to_features_scaled(train_df: pd.DataFrame, seq_len: int = globals.SEQ_LEN, val_df: pd.DataFrame = None, stride: int = globals.STRIDE):
    def df_to_seq(df: pd.DataFrame):
        try:
            X = []
            n_rows = len(df)

            for i in range(0, n_rows - seq_len + 1, stride):
                seq = df.iloc[i:i + seq_len][globals.FEACTURE].values
                scaler = StandardScaler()
                seq_scaled = scaler.fit_transform(seq)  # 시퀀스 단위 스케일링
                X.append(seq_scaled)

            if len(X) == 0:
                return np.empty((0, seq_len, len(globals.FEACTURE)), dtype=np.float32)
            return np.asarray(X, dtype=np.float32)
        except Exception as e:
            print("Error occurred in df_to_seq:")
            print(e)                   # 에러 메시지
            traceback.print_exc()      # 전체 traceback 출력
            return np.empty((0, seq_len, len(globals.FEACTURE)), dtype=np.float32)

    X_train = df_to_seq(train_df)
    X_val = df_to_seq(val_df) if val_df is not None else None

    return X_train, X_val

def train_start(train_dataset, val_dataset, batch_size=32, epochs=50, lr=0.0005, device=None, stop_event=None, patience=10):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    # 모델 초기화
    model = CnnDenseLstm(
        input_size=len(globals.FEACTURE), 
        lstm_hidden_size=128, 
        lstm_layers=3, 
        dropout=0.3
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0  # 개선 없는 epoch 수

    for epoch in range(epochs):
        if stop_event and stop_event.is_set():
            print("학습 중지 이벤트 발생")
            break

        # ===== Train =====
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_train_loss = total_loss / len(train_dataset)

        # ===== Validation =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        avg_val_loss = val_loss / len(val_dataset)

        # Best 모델 저장 및 Early Stopping 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # 초기화
            torch.save(model.state_dict(), save_path)
            print(f"[Model Saved] Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Val loss 개선 없음 {patience} epoch. 학습 중지 이벤트 발생")
                if stop_event:
                    stop_event.set()
                break

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print(f"최종 Best Val Loss: {best_val_loss:.6f}, 모델 저장 위치: {save_path}")

def main(stop_event=None, seq_len=globals.SEQ_LEN, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print("setting")
    print(f"device : {device}")    
    print(f"SEQ_LEN : {globals.SEQ_LEN}")
    print(f"STRIDE : {globals.STRIDE}")

    # ===== 데이터 읽기 =====

    if globals.Recorder == "postgres":
        user_all: list[MousePoint] = DBController.read(True)
        macro_all: list[MacroMousePoint] = DBController.read(False)
        n = min(len(user_all), len(macro_all))
        user_points = user_all[:n]
        macro_points = macro_all[:n]
        
        user_df_chunk = pd.DataFrame({
            "timestamp": [p.timestamp for p in user_points],
            "x": [p.x for p in user_points],
            "y": [p.y for p in user_points],
        })
        macro_df_chunk = pd.DataFrame({
            "timestamp": [p.timestamp for p in macro_points],
            "x": [p.x for p in macro_points],
            "y": [p.y for p in macro_points],
        })        
    elif globals.Recorder == "json":
        user_all: list[dict] = JsonController.read(True)
        macro_all: list[dict] = JsonController.read(False)

        n = min(len(user_all), len(macro_all))
        user_points = user_all[:n]      # <- dict 타입 선언 X, 그냥 변수에 할당
        macro_points = macro_all[:n]

        user_df_chunk = pd.DataFrame({
            "timestamp": [p.get("timestamp") for p in user_points],
            "x": [p.get("x") for p in user_points],
            "y": [p.get("y") for p in user_points],
        })
        macro_df_chunk = pd.DataFrame({
            "timestamp": [p.get("timestamp") for p in macro_points],
            "x": [p.get("x") for p in macro_points],
            "y": [p.get("y") for p in macro_points],
        })

    # ===== Feature 계산 =====
    setting_user_df_chunk: pd.DataFrame = indicators_generation(user_df_chunk)
    setting_macro_df_chunk: pd.DataFrame = indicators_generation(macro_df_chunk)
    
    setting_user_df_chunk = setting_user_df_chunk.sort_values('timestamp').reset_index(drop=True)
    setting_macro_df_chunk = setting_macro_df_chunk.sort_values('timestamp').reset_index(drop=True)

    # ===== Feature 필터 =====
    setting_user_df_chunk = setting_user_df_chunk[globals.FEACTURE].copy()
    setting_macro_df_chunk = setting_macro_df_chunk[globals.FEACTURE].copy()


    if len(user_points) < seq_len or len(macro_points) < seq_len:
        print("데이터가 충분하지 않습니다.")
        return

    # ===== Train/Val split =====
    user_train_df, user_val_df = train_test_split(setting_user_df_chunk, test_size=0.2, shuffle=False)
    macro_train_df, macro_val_df = train_test_split(setting_macro_df_chunk, test_size=0.2, shuffle=False)

    # ===== Sequence =====q
    user_train_seq, user_val_seq = points_to_features_scaled(train_df=user_train_df, val_df=user_val_df, seq_len=globals.SEQ_LEN, stride=globals.STRIDE)
    macro_train_seq, macro_val_seq = points_to_features_scaled(train_df=macro_train_df, val_df=macro_val_df, seq_len=globals.SEQ_LEN, stride=globals.STRIDE)

    # ===== 입력 + 라벨 결합 =====
    X_train = np.concatenate([user_train_seq, macro_train_seq], axis=0)
    y_train = np.concatenate([
        np.ones(len(user_train_seq)),
        np.zeros(len(macro_train_seq))
    ])

    X_val = np.concatenate([user_val_seq, macro_val_seq], axis=0)
    y_val = np.concatenate([
        np.ones(len(user_val_seq)),
        np.zeros(len(macro_val_seq))
    ])

    # ===== Dataset 생성 =====
    train_dataset = MouseFeatureDataset(X_train, y_train)
    val_dataset   = MouseFeatureDataset(X_val, y_val)

    print(f"Train seq: {len(X_train)}, Val seq: {len(X_val)}")
    print(f"Input shape: {X_train.shape}")

    # ===== 학습 시작 =====
    train_start(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        stop_event=stop_event
    )
