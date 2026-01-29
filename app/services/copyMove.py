import os
import json

from pynput.mouse import Controller, Button, Listener
from pynput import mouse

from multiprocessing import Event

from datetime import datetime

import time

import app.core.globals as globals
from multiprocessing import Queue
import app.repostitories.DBController as DBController
import app.repostitories.JsonController as JsonController
from app.services.cunsume_q import cunsume_q


def start_mouse_click_listener(stop_event):
    """마우스 클릭 이벤트 기록용 리스너"""
    def on_click(x, y, button, pressed):
        if stop_event.is_set():
            return False  # 리스너 종료

        event_type = 1 if pressed else 2
        globals.IS_PRESSED = 1 if pressed else 0

        data = {
            'timestamp': datetime.now().isoformat(),
            'x': int(x),
            'y': int(y),
            'event_type': event_type,
            'is_pressed': globals.IS_PRESSED
        }

        globals.MOUSE_QUEUE.put(data)

    listener = Listener(on_click=on_click)
    listener.start()
    return listener


def copy_move(stop_event=None, log_queue: Queue = None):
    """마우스 좌표 + 클릭 재생 (is_pressed 기반 눌림 유지)"""
    if stop_event is None:
        stop_event = Event()

    mouse = Controller()
    if log_queue:
        log_queue.put("[Process] 마우스 재생 시작")

    # 기록 데이터 읽기
    data = None
    if globals.Recorder == "json":
        data = JsonController.read(user=True, log_queue=globals.LOG_QUEUE)
    elif globals.Recorder == "postgres":
        data = DBController.read(user=True, log_queue=globals.LOG_QUEUE)

    if not data:
        if log_queue:
            log_queue.put("[Process] 재생할 데이터 없음")
        return

    start = True
    prev_pressed = 0  # 이전 눌림 상태

    for p in data:
        if stop_event.is_set():
            break

        # dict / ORM 객체 대응
        if isinstance(p, dict):
            x = p.get("x", 0)
            y = p.get("y", 0)
            timestamp = datetime.fromisoformat(p.get("timestamp", "1970-01-01T00:00:00"))
            is_pressed = p.get("is_pressed", 0)
        else:
            x = getattr(p, "x", 0)
            y = getattr(p, "y", 0)
            timestamp = datetime.fromisoformat(getattr(p, "timestamp", "1970-01-01T00:00:00"))
            is_pressed = getattr(p, "is_pressed", 0)

        if start:
            start = False
            prev_x, prev_y = x, y
            prev_timestamp = timestamp
            # 첫 좌표 이동
            mouse.position = (x, y)
            # 첫 눌림 처리
            if is_pressed == 1:
                mouse.press(Button.left)
            prev_pressed = is_pressed
            continue

        # 이동 간격 계산
        interval = (timestamp - prev_timestamp).total_seconds()
        steps = max(int(interval / 0.005), 1)  # 5ms 단위 보간
        t0 = time.perf_counter()

        for step in range(1, steps + 1):
            if stop_event.is_set():
                break

            # 선형 보간
            xi = prev_x + (x - prev_x) * (step / steps)
            yi = prev_y + (y - prev_y) * (step / steps)
            mouse.position = (xi, yi)

            # 시간 맞추기
            while True:
                now = time.perf_counter()
                t_target = t0 + interval * (step / steps)
                if now >= t_target:
                    break

        # is_pressed 기준 클릭 유지
        if prev_pressed == 0 and is_pressed == 1:
            mouse.press(Button.left)
        elif prev_pressed == 1 and is_pressed == 0:
            mouse.release(Button.left)

        prev_pressed = is_pressed
        prev_x, prev_y = x, y
        prev_timestamp = timestamp

    # 마지막 상태 체크: 눌려있으면 떼기
    if prev_pressed == 1:
        mouse.release(Button.left)

    if log_queue:
        log_queue.put("[Process] 마우스 재생 완료")
    stop_event.set()
