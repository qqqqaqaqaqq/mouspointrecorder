import os
import json

import pyautogui
from pynput import mouse

from multiprocessing import Event

from datetime import datetime

import time

import app.core.globals as globals
from app.repostitories.DBController import SessionLocal, MousePoint
from multiprocessing import Queue

from app.services.cunsume_q import cunsume_q

pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

def start_mouse_click_listener(stop_event):
    def on_click(x, y, button, pressed):
        if stop_event.is_set():
            return False  # 리스너 종료

        event_type = 1 if pressed else 2
        globals.IS_PRESSED = 1 if pressed else 0

        data = {
            'timestamp': datetime.now(),
            'x': int(x),
            'y': int(y),
            'event_type': event_type,
            'is_pressed': globals.IS_PRESSED
        }

        globals.MOUSE_QUEUE.put(data)

    listener = mouse.Listener(on_click=on_click)
    listener.start()
    return listener

def record_mouse_path(stop_event = None, record=True, interval=0.005, log_queue:Queue=None):
    if stop_event is None:
        stop_event = Event()

    # setting
    globals.IS_PRESSED = 0

    log_queue.put("[Process] 마우스 경로 생성 시작")

    click_listener = start_mouse_click_listener(stop_event)

    while not stop_event.is_set():
        x, y = pyautogui.position()
        timestamp = datetime.now()

        data = {
            'timestamp': timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': 0,
            'is_pressed': globals.IS_PRESSED
        }

        if record:
            globals.MOUSE_QUEUE.put(data)

        if globals.MOUSE_QUEUE.qsize() >= globals.MAX_QUEUE_SIZE:
            log_queue.put("Data 5000개 초과.. 저장 중..")
            # isUser => True
            cunsume_q(record=record, isUser=True, log_queue=log_queue)
            log_queue.put("저장 완료 다음 시퀀스 준비")

        time.sleep(interval)

    click_listener.stop()

    # 남아있는거 PUSH 후 종료
    cunsume_q(record=record, isUser=True, log_queue=log_queue)

    stop_event.set()