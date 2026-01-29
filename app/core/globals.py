import queue
from app.core.settings import settings

MOUSE_QUEUE = queue.Queue()
IS_PRESSED = 0
MAX_QUEUE_SIZE = 5000

SEQ_LEN = settings.SEQ_LEN
STRIDE = settings.STRIDE

FEACTURE = [
    "speed",
    "jerk",
    "turn",
    "turn_acc",
    "acc_change",
    "jerk_change",
    # "event_type",
    # "event_up",
]

MACRO_DETECTOR  = [] 

Recorder = settings.Recorder
JsonPath = settings.JsonPath

threshold = settings.threshold

# model
lstm_hidden_size=settings.lstm_hidden_size
lstm_layers=settings.lstm_layers
dropout=settings.dropout
batch_size=settings.batch_size
ir=settings.ir

LOG_QUEUE = None

def init_manager():
    global LOG_QUEUE
    from multiprocessing import Manager
    manager = Manager()
    LOG_QUEUE = manager.Queue()