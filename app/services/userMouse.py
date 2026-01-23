import pyautogui

import time
from datetime import datetime

import app.core.globals as globals

pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

def record_mouse_path(interval=0.01, stop_event=None):
    while True:
        if stop_event and stop_event.is_set():
            print("마우스 기록 중지")
            break

        x, y = pyautogui.position()
        timestamp = datetime.now()
        data = {'time': timestamp, 'x': x, 'y': y}

        globals.MOUSE_QUEUE.put(data)

        time.sleep(interval)
