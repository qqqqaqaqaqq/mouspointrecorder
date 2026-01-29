import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df:pd.DataFrame = df_chunk.copy()

    # 시간 정렬
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    epsilon = 1e-6  # 0 나누기 방지

    # 시간 차 (초)
    df["timestamp"] = pd.to_datetime(df["timestamp"]) 
    df["dt"] = df["timestamp"].diff().dt.total_seconds()
    df["dt"] = df["dt"].clip(lower=0.001)

    # 위치 변화량
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()

    # 이동 거리
    df["dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    # 속도
    df["speed"] = df["dist"] / df["dt"]

    # 로그 속도 (분포 안정화)
    df["speed_log"] = np.log1p(df["speed"])

    # 가속도
    df["acc"] = df["speed"].diff()
    
    df["acc_change"] = abs((df["acc"] - df["acc"].shift(1)) / (df["acc"].shift(1) + epsilon))

    # 로그 가속도 (부호 유지)
    df["acc_log"] = np.sign(df["acc"]) * np.log1p(np.abs(df["acc"]))

    # jerk (🔥 매우 중요)
    df["jerk"] = df["acc"].diff()

    df["jerk_change"] = abs((df["jerk"] - df["jerk"].shift(1)) / (df["jerk"].shift(1) + epsilon))

    # 이동 각도
    df["angle"] = np.arctan2(df["dy"], df["dx"])

    # 방향 변화량
    df["turn"] = df["angle"].diff()

    # 각도 wrap 보정 (-pi ~ pi)
    df["turn"] = (df["turn"] + np.pi) % (2 * np.pi) - np.pi

    # 방향 가속도 (🔥 매크로 잘 잡힘)
    df["turn_acc"] = df["turn"].diff()
    
    df["event_down"] = (df["event_type"] == 1).astype(int)
    df["event_up"]   = (df["event_type"] == 2).astype(int)

    df["press_duration"] = 0.0
    pressed = False
    start_time = None

    for i in range(len(df)):
        if df.loc[i, "event_type"] == 1:  # down
            pressed = True
            start_time = df.loc[i, "timestamp"]

        elif df.loc[i, "event_type"] == 2 and pressed:  # up
            pressed = False
            if start_time is not None:
                # 다운~업 사이 누른 시간 계산
                df.loc[i, "press_duration"] = (
                    df.loc[i, "timestamp"] - start_time
                ).total_seconds()
            start_time = None


    # NaN / inf → 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
