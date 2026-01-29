import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import traceback
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import app.core.globals as globals

# 모델
from app.models.MousePoint import MousePoint, MacroMousePoint
from app.models.CnnDenseLstm import CnnDenseLstm, CnnDenseLstmOneClass
from app.models.MouseFeatureDataset import MouseFeatureDataset, MessMouseFeatureDataset

import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController

from app.services.indicators import indicators_generation
from multiprocessing import Queue

save_path = "app/models/weights/mouse_macro_lstm_best.pt"


# stride = > seq 데이터 겹치는 양 (How much each sequence shifts forward)
def points_to_features_scaled(df_chunk: pd.DataFrame, seq_len: int = globals.SEQ_LEN, stride: int = globals.STRIDE, log_queue:Queue=None):
    Total_Pass_SEQ = 0
    
    # 시퀀스내의 스탭들이 모드 동일할 때 제거 => 움직임이 없을 경우 제외
    def df_to_seq(df: pd.DataFrame):
        nonlocal Total_Pass_SEQ
        try:
            X = []
            n_rows = len(df)

            for i in range(0, n_rows - seq_len + 1, stride):
                seq = df.iloc[i:i + seq_len][globals.FEACTURE].values

                if np.all(seq==0):
                    continue

                scalar = StandardScaler()
                seq_scalar = scalar.fit_transform(seq)

                Total_Pass_SEQ += 1
                X.append(seq_scalar)

            if len(X) == 0:
                return np.empty((0, seq_len, len(globals.FEACTURE)), dtype=np.float32)
            return np.asarray(X, dtype=np.float32)
        
        except Exception as e:
            log_queue.put("Error occurred in df_to_seq:")
            log_queue.put(e)                   
            traceback.print_exc()      
            return np.empty((0, seq_len, len(globals.FEACTURE)), dtype=np.float32)

    data = df_to_seq(df_chunk)

    return data, Total_Pass_SEQ

# train
def train_start(train_dataset, val_dataset, batch_size=globals.batch_size, epochs=100, lr=globals.ir, device=None, model=None, stop_event=None, patience=20, log_queue:Queue=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # labelling
    criterion = nn.BCEWithLogitsLoss()

    # oneclass
    # criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0  # 개선 없는 epoch 수

    for epoch in range(epochs):
        if stop_event and stop_event.is_set():
            log_queue.put("학습 중지 이벤트 발생")
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
            log_queue.put(f"[Model Saved] Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")

        else:  
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_queue.put(f"Val loss 개선 없음 {patience} epoch. 학습 중지 이벤트 발생")
                if stop_event:
                    stop_event.set()
                break
        
        log_queue.put(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    log_queue.put(f"최종 Best Val Loss: {best_val_loss:.6f}, 모델 저장 위치: {save_path}")


# oneclass
# def main(stop_event=None, seq_len=globals.SEQ_LEN, device=None, log_queue:Queue=None):
#     device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

#     log_queue.put("setting")
#     log_queue.put(f"device : {device}")    
#     log_queue.put(f"SEQ_LEN : {globals.SEQ_LEN}")
#     log_queue.put(f"STRIDE : {globals.STRIDE}")

#     # ===== 데이터 읽기 =====

#     if globals.Recorder == "postgres":
#         user_all: list[MousePoint] = DBController.read(True, log_queue=log_queue)

#         log_queue.put(f"data 길이 {len(user_all)}")

#         user_points:list[MousePoint] = user_all[:len(user_all)]

#         user_df_chunk = pd.DataFrame({
#             "timestamp": [p.timestamp for p in user_points],
#             "x": [p.x for p in user_points],
#             "y": [p.y for p in user_points],
#             "event_type" : [p.event_type for p in user_points],
#             "is_pressed" : [p.is_pressed for p in user_points],            
#         })      
#     elif globals.Recorder == "json":
#         user_all: list[dict] = JsonController.read(True, log_queue=log_queue)
#         macro_all: list[dict] = JsonController.read(False, log_queue=log_queue)

#         n = min(len(user_all), len(macro_all))

#         log_queue.put(f"user_all length : {len(user_all)}")
#         log_queue.put(f"macro_all length : {len(macro_all)}")
#         log_queue.put(f"data 길이 {n}")

#         user_points:list[dict] = user_all[:n]      # <- dict 타입 선언 X, 그냥 변수에 할당

#         user_df_chunk = pd.DataFrame({
#             "timestamp": [p.get("timestamp") for p in user_points],
#             "x": [p.get("x") for p in user_points],
#             "y": [p.get("y") for p in user_points],
#             "event_type" : [p.get("event_type") for p in user_points],
#             "is_pressed" : [p.get("is_pressed") for p in user_points],                
#         })

#     # ===== Feature 계산 =====
#     setting_user_df_chunk: pd.DataFrame = indicators_generation(user_df_chunk)

#     setting_user_df_chunk = setting_user_df_chunk.sort_values('timestamp').reset_index(drop=True)

#     # ===== Feature 필터 =====
#     setting_user_df_chunk = setting_user_df_chunk[globals.FEACTURE].copy()

#     if len(user_points) < seq_len:
#         log_queue.put("데이터가 충분하지 않습니다.")
#         return

#     # ===== Train/Val split =====
#     user_train_df: pd.DataFrame
#     user_val_df: pd.DataFrame

#     user_train_df, user_val_df = train_test_split(setting_user_df_chunk, test_size=0.2, shuffle=False)

#     # ===== Sequence =====q
#     user_train_seq, user_val_seq = points_to_features_scaled(
#         train_df=user_train_df, 
#         val_df=user_val_df, 
#         seq_len=globals.SEQ_LEN, 
#         stride=globals.STRIDE,
#         log_queue=log_queue
#         )

#     if len(user_train_seq) == 0 or len(user_val_seq) == 0:
#         log_queue.put("시퀀스 길이가 충분하지 않아 학습 불가")
#         return

#     train_dataset = MessMouseFeatureDataset(user_train_seq)
#     val_dataset   = MessMouseFeatureDataset(user_val_seq)

#     # 모델 초기화
#     model = CnnDenseLstmOneClass(
#         input_size=len(globals.FEACTURE), 
#         lstm_hidden_size=globals.lstm_hidden_size, 
#         lstm_layers=globals.lstm_layers, 
#         dropout=globals.dropout
#     ).to(device)

#     # ===== 학습 시작 =====
#     train_start(
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         device=device,
#         model=model,
#         stop_event=stop_event,
#         log_queue=log_queue
#     )



def compress_zero_sum_rows(df: pd.DataFrame, max_zeros=5):
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

# labeling
def main(stop_event=None, seq_len=globals.SEQ_LEN, device=None, log_queue:Queue=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    log_queue.put("setting")
    log_queue.put(f"device : {device}")    
    log_queue.put(f"SEQ_LEN : {globals.SEQ_LEN}")
    log_queue.put(f"STRIDE : {globals.STRIDE}")

    # ===== 데이터 읽기 =====

    if globals.Recorder == "postgres":
        user_all: list[MousePoint] = DBController.read(True, log_queue=log_queue)
        macro_all: list[MacroMousePoint] = DBController.read(False, log_queue=log_queue)
        n = min(len(user_all), len(macro_all))

        log_queue.put(f"user_all length : {len(user_all)}")
        log_queue.put(f"macro_all length : {len(macro_all)}")
        log_queue.put(f"data 길이 {n}")

        user_points:list[MousePoint] = user_all[:n]
        macro_points:list[MacroMousePoint] = macro_all[:n]
        
        user_df_chunk = pd.DataFrame({
            "timestamp": [p.timestamp for p in user_points],
            "x": [p.x for p in user_points],
            "y": [p.y for p in user_points],
            "event_type" : [p.event_type for p in user_points],
            "is_pressed" : [p.is_pressed for p in user_points],            
        })
        macro_df_chunk = pd.DataFrame({
            "timestamp": [p.timestamp for p in macro_points],
            "x": [p.x for p in macro_points],
            "y": [p.y for p in macro_points],
            "event_type" : [p.event_type for p in macro_points],
            "is_pressed" : [p.is_pressed for p in macro_points],                
        })        
    elif globals.Recorder == "json":
        user_all: list[dict] = JsonController.read(True, log_queue=log_queue)
        macro_all: list[dict] = JsonController.read(False, log_queue=log_queue)

        n = min(len(user_all), len(macro_all))

        log_queue.put(f"user_all length : {len(user_all)}")
        log_queue.put(f"macro_all length : {len(macro_all)}")
        log_queue.put(f"data 길이 {n}")

        user_points:list[dict] = user_all[:n]      # <- dict 타입 선언 X, 그냥 변수에 할당
        macro_points:list[dict] = macro_all[:n]

        user_df_chunk = pd.DataFrame({
            "timestamp": [p.get("timestamp") for p in user_points],
            "x": [p.get("x") for p in user_points],
            "y": [p.get("y") for p in user_points],
            "event_type" : [p.get("event_type") for p in user_points],
            "is_pressed" : [p.get("is_pressed") for p in user_points],                
        })
        macro_df_chunk = pd.DataFrame({
            "timestamp": [p.get("timestamp") for p in macro_points],
            "x": [p.get("x") for p in macro_points],
            "y": [p.get("y") for p in macro_points],
            "event_type" : [p.get("event_type") for p in macro_points],
            "is_pressed" : [p.get("is_pressed") for p in macro_points],               
        })

    # ===== Feature 계산 =====
    setting_user_df_chunk: pd.DataFrame = indicators_generation(user_df_chunk)
    setting_macro_df_chunk: pd.DataFrame = indicators_generation(macro_df_chunk)
    
    # 문제 되는 int 값 인덱스
    bad_idx = setting_user_df_chunk[
        setting_user_df_chunk['timestamp'].apply(lambda x: isinstance(x, int))
    ].index

    for idx in bad_idx:
        # -1 ~ +1 범위 행 출력 (주의: 범위 벗어나면 안전하게 처리)
        start = max(idx - 1, 0)
        end = min(idx + 1, len(setting_user_df_chunk) - 1)
        print(f"\n문제 행 인덱스 {idx} 주변 행:")
        print(setting_user_df_chunk.loc[start:end])


    bad_idx = setting_macro_df_chunk[
        setting_macro_df_chunk['timestamp'].apply(lambda x: isinstance(x, int))
    ].index

    for idx in bad_idx:
        # -1 ~ +1 범위 행 출력 (주의: 범위 벗어나면 안전하게 처리)
        start = max(idx - 1, 0)
        end = min(idx + 1, len(setting_macro_df_chunk) - 1)
        print(f"\n문제 행 인덱스 {idx} 주변 행:")
        print(setting_macro_df_chunk.loc[start:end])


    setting_user_df_chunk = setting_user_df_chunk.sort_values('timestamp').reset_index(drop=True)
    setting_macro_df_chunk = setting_macro_df_chunk.sort_values('timestamp').reset_index(drop=True)

    # ===== Feature 필터 =====
    setting_user_df_chunk = setting_user_df_chunk[globals.FEACTURE].copy()
    setting_macro_df_chunk = setting_macro_df_chunk[globals.FEACTURE].copy()

    # ===== 0 압축 =====
    setting_user_df_chunk = compress_zero_sum_rows(setting_user_df_chunk, max_zeros=1)
    setting_macro_df_chunk = compress_zero_sum_rows(setting_macro_df_chunk, max_zeros=1)

    print(f"setting_user_df_chunk : {setting_user_df_chunk}")
    print(f"setting_macro_df_chunk : {setting_macro_df_chunk}")

    if len(user_points) < seq_len or len(macro_points) < seq_len:
        log_queue.put("데이터가 충분하지 않습니다.")
        return

    # ===== Train/Val split =====
    user_train_df: pd.DataFrame
    user_val_df: pd.DataFrame
    macro_train_df: pd.DataFrame
    macro_val_df:pd.DataFrame

    user_train_df, user_val_df = train_test_split(setting_user_df_chunk, test_size=0.2, shuffle=False)
    macro_train_df, macro_val_df = train_test_split(setting_macro_df_chunk, test_size=0.2, shuffle=False)
    
    # ===== Sequence =====q
    user_train_seq, user_train_pass_seq = points_to_features_scaled(
        df_chunk=user_train_df, 
        seq_len=globals.SEQ_LEN, 
        stride=globals.STRIDE,
        log_queue=log_queue
        )

    user_val_seq, user_val_pass_seq = points_to_features_scaled(
        df_chunk=user_val_df, 
        seq_len=globals.SEQ_LEN, 
        stride=globals.STRIDE,
        log_queue=log_queue
        )

    macro_train_seq, macro_train_pass_seq = points_to_features_scaled(
        df_chunk=macro_train_df, 
        seq_len=globals.SEQ_LEN, 
        stride=globals.STRIDE,
        log_queue=log_queue
        )

    macro_val_seq, macro_val_pass_seq = points_to_features_scaled(
        df_chunk=macro_val_df, 
        seq_len=globals.SEQ_LEN, 
        stride=globals.STRIDE,
        log_queue=log_queue
        )

    traindata_length = min(user_train_pass_seq, macro_train_pass_seq)
    valdata_length = min(user_val_pass_seq, macro_val_pass_seq)

    user_train_seq = user_train_seq[:traindata_length]
    user_val_seq = user_val_seq[:valdata_length]
    macro_train_seq = macro_train_seq[:traindata_length]
    macro_val_seq = macro_val_seq[:valdata_length]
    
    log_queue.put(f"Length => train data : {traindata_length}, val data : {valdata_length}")
    
    if len(user_train_seq) == 0 or len(user_val_seq) == 0:
        log_queue.put("시퀀스 길이가 충분하지 않아 학습 불가")
        return

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

    log_queue.put(f"Train seq: {len(X_train)}, Val seq: {len(X_val)}")
    log_queue.put(f"Input shape: {X_train.shape}")

    # 모델 초기화
    model = CnnDenseLstm(
        input_size=len(globals.FEACTURE), 
        lstm_hidden_size=globals.lstm_hidden_size, 
        lstm_layers=globals.lstm_layers, 
        dropout=globals.dropout
    ).to(device)

    if stop_event and stop_event.is_set():
        log_queue.put("학습 중지 완료")
        return
    
    # ===== 학습 시작 =====
    train_start(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        model=model,
        stop_event=stop_event,
        log_queue=log_queue
    )
