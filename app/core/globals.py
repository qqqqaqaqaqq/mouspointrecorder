import queue
from app.core.settings import settings

MOUSE_QUEUE = queue.Queue()
IS_PRESSED = 0
MAX_QUEUE_SIZE = 5000

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEACTURE = [
    "dist",
    "speed",
    "acc",
    "jerk",
    "turn",
    "turn_acc",
    "event_down",
    "event_up",
    "is_pressed",
    "press_duration"
]

MACRO_DETECTOR  = [] 

Recorder = settings.Recorder
JsonPath = settings.JsonPath

threshold = settings.threshold


# model
lstm_hidden_size=32
lstm_layers=2
dropout=0.4

LOG_QUEUE = None


def init_manager():
    global LOG_QUEUE
    from multiprocessing import Manager
    manager = Manager()
    LOG_QUEUE = manager.Queue()