import queue
from app.core.settings import settings
from collections import deque

MOUSE_QUEUE = queue.Queue()

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEACTURE = [
    "dist",       # 이동 거리
    "speed",      # 속도
    "acc",        # 가속도
    "jerk",       # 가속도 변화량
    "turn",       # 방향 변화량
    "turn_acc"    # 방향 가속도
]

MACRO_DETECTOR  = []
