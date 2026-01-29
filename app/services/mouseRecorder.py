from pynput.mouse import Controller
from pynput import mouse

from multiprocessing import Event

from datetime import datetime

import time

import app.core.globals as globals
from multiprocessing import Queue

from app.services.cunsume_q import cunsume_q

def start_mouse_click_listener(stop_event):
    def on_click(x, y, button, pressed):
        if globals.Recorder == "postgres":
            record_timestamp = datetime.now()
        elif globals.Recorder == "json":
            record_timestamp = datetime.now().isoformat()      

        if stop_event.is_set():
            return False  # 리스너 종료

        event_type = 1 if pressed else 2
        globals.IS_PRESSED = 1 if pressed else 0

        data = {
            'timestamp': record_timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': event_type,
            'is_pressed': globals.IS_PRESSED
        }

        globals.MOUSE_QUEUE.put(data)

    listener = mouse.Listener(on_click=on_click)
    listener.start()
    return listener

def record_mouse_path(isUser, stop_event = None, record=True, interval=0.005, log_queue:Queue=None):
    if stop_event is None:
        stop_event = Event()

    mouse = Controller()

    # setting
    globals.IS_PRESSED = 0

    log_queue.put("[Process] 마우스 경로 생성 시작")
    i = 1
    click_listener = start_mouse_click_listener(stop_event)
    
    tolerance = 0.00005

    pre_timestamp = None
    start = time.perf_counter() 
    while not stop_event.is_set():  
        x, y = mouse.position
        timestamp = datetime.now().timestamp()
        if globals.Recorder == "postgres":
            record_timestamp = datetime.now()
        elif globals.Recorder == "json":
            record_timestamp = datetime.now().isoformat()            

        if pre_timestamp is not None:
            if (timestamp - pre_timestamp) < tolerance:
                continue 
        
        pre_timestamp = timestamp

        data = {
            'timestamp': record_timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': 0,
            'is_pressed': globals.IS_PRESSED
        }

        if record:
            globals.MOUSE_QUEUE.put(data)

        if globals.MOUSE_QUEUE.qsize() >= globals.MAX_QUEUE_SIZE:
            log_queue.put(f"Data 5000개 초과.. 누적 {5000 * i}")
            i += 1
            # isUser => True
            cunsume_q(record=record, isUser=isUser, log_queue=log_queue)
            log_queue.put("저장 완료 다음 시퀀스 준비")

        end = time.perf_counter()  # 루프 끝 시간
        loop_time = end - start
        # print(f"[Loop Timing] 한 사이클 소요: {loop_time:.6f}초")        
        
        start = time.perf_counter()

    click_listener.stop()

    # 남아있는거 PUSH 후 종료
    cunsume_q(record=record, isUser=isUser, log_queue=log_queue)

    stop_event.set()