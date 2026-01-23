# app/train/mouse_lstm_feature_train_val.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from app.models.MousePoint import MousePoint, MacroMousePoint
from app.models.DenseLstm import DenseLSTMModel
from app.repostitories.DBController import read

save_path = "app/models/weights/mouse_macro_lstm_best.pt"

# 논리
# gpt 피셜 좌표값 기준으로 하면 fuck 된다.
# 그러니 feacture를 생성해서 해라
# ㅇㅋ 하고 바꿈 시불련 ㅈㄴ 똑똑해.
def points_to_features_minmax(points: list[MousePoint], seq_len=15):
    if len(points) < 2:
        return np.empty((0, seq_len, 6), dtype=np.float32)

    # 좌표 min-max
    x_values = [p.x for p in points]
    y_values = [p.y for p in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    data = []
    prev = points[0]
    prev_speed = 0.0

    for p in points[1:]:
        dt = (p.timestamp - prev.timestamp).total_seconds()
        dx = (p.x - prev.x) / (x_max - x_min + 1e-6)
        dy = (p.y - prev.y) / (y_max - y_min + 1e-6)
        speed = np.sqrt(dx**2 + dy**2) / (dt + 1e-6)
        acc = speed - prev_speed
        angle = np.arctan2(dy, dx)

        data.append([dt, dx, dy, speed, acc, angle])

        prev = p
        prev_speed = speed

    data = np.array(data, dtype=np.float32)

    # 시퀀스로 변환
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])

    return np.array(sequences, dtype=np.float32)

class MouseFeatureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def main(stop_event=None, seq_len=100, batch_size=32, epochs=20, lr=0.0005, device=None, val_ratio=0.2):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # DB에서 데이터 읽기 (각 클래스 1500개)
    user_points: list[MousePoint] = read(True)[:1500]
    macro_points: list[MacroMousePoint] = read(False)[:1500]

    if len(user_points) < seq_len or len(macro_points) < seq_len:
        print("데이터가 충분하지 않습니다.")
        return

    # 시퀀스 변환
    user_seq = points_to_features_minmax(user_points, seq_len)
    macro_seq = points_to_features_minmax(macro_points, seq_len)

    # 입력과 라벨
    X = np.concatenate([user_seq, macro_seq], axis=0)
    y = np.concatenate([np.ones(len(user_seq)), np.zeros(len(macro_seq))], axis=0)

    print(f"총 시퀀스 수: {len(X)}, 입력 shape: {X.shape}")

    # Dataset / Train-Val Split
    dataset = MouseFeatureDataset(X, y)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 초기화 (Dropout 적용)
    model = DenseLSTMModel(input_size=6, hidden_size=128, num_layers=3, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    # 학습
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

        # ===== Best 모델 저장 =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Model Saved] Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print(f"최종 Best Val Loss: {best_val_loss:.6f}, 모델 저장 위치: {save_path}")