import queue
from app.core.settings import settings

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

Recorder = settings.Recorder
JsonPath = settings.JsonPath

LOG_QUEUE = None


def init_manager():
    global LOG_QUEUE
    from multiprocessing import Manager
    manager = Manager()
    LOG_QUEUE = manager.Queue()