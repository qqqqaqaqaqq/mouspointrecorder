import pandas as pd
import numpy as np

def indicators_generation(df_chunk: pd.DataFrame) -> pd.DataFrame:
    df = df_chunk.copy()

    # 시간 정렬
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 시간 차 (초)
    df["dt"] = df["timestamp"].diff().dt.total_seconds()
    df.loc[df["dt"] <= 0, "dt"] = np.nan

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

    # 로그 가속도 (부호 유지)
    df["acc_log"] = np.sign(df["acc"]) * np.log1p(np.abs(df["acc"]))

    # jerk (🔥 매우 중요)
    df["jerk"] = df["acc"].diff()

    # 이동 각도
    df["angle"] = np.arctan2(df["dy"], df["dx"])

    # 방향 변화량
    df["turn"] = df["angle"].diff()

    # 각도 wrap 보정 (-pi ~ pi)
    df["turn"] = (df["turn"] + np.pi) % (2 * np.pi) - np.pi

    # 방향 가속도 (🔥 매크로 잘 잡힘)
    df["turn_acc"] = df["turn"].diff()

    # 방향 벡터
    df["sin"] = np.sin(df["angle"])
    df["cos"] = np.cos(df["angle"])

    # NaN / inf → 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
