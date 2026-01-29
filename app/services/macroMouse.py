import os
import json
from multiprocessing import Event
from collections import deque

import pyautogui
from pynput import mouse

import time, random, math
from datetime import datetime
import app.core.globals as globals
from multiprocessing import Queue
from app.services.cunsume_q import cunsume_q

pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

# ----------------- 이동 패턴 함수 -----------------
def catmull_rom_spline(p0, p1, p2, p3, t):
    t2 = t*t
    t3 = t2*t
    x = 0.5*((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
    y = 0.5*((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    return x, y

def ease_in_out_quad_random(t):
    # 가속/감속 계수를 랜덤으로 조금 변형
    accel = 1 + random.uniform(-0.2, 0.2)  # 가속 구간 변화 ±20%
    decel = 1 + random.uniform(-0.2, 0.2)  # 감속 구간 변화 ±20%

    if t < 0.5:
        return 2 * accel * t * t
    else:
        return -1 + (4 - 2 * decel * t) * t
def linear(t): return t
def ease_out_cubic(t): return 1 - pow(1-t, 3)
def ease_in_out_s_curve(t): return t*t*(3 - 2*t)

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

def macro_click(x, y, move_mouse=False):
    # click down

    if globals.Recorder == "postgres":
        record_timestamp = datetime.now()
    elif globals.Recorder == "json":
        record_timestamp = datetime.now().isoformat()   

    if move_mouse:
        pyautogui.mouseDown(x=x, y=y)
    else:
        globals.IS_PRESSED = 1
        globals.MOUSE_QUEUE.put({
            'timestamp': record_timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': 1,
            'is_pressed': 1
        })

    time.sleep(random.uniform(0.05, 0.15))

    # click up
    if move_mouse:
        pyautogui.mouseUp(x=x, y=y)
    else:
        globals.IS_PRESSED = 0
        globals.MOUSE_QUEUE.put({
            'timestamp': record_timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': 2,
            'is_pressed': 0
        })

def record_mouse_path(stop_event = None, move_mouse=False, record=True, interval=0.005, log_queue:Queue=None):
    if stop_event is None:
        stop_event = Event()

    # setting
    globals.IS_PRESSED = 0

    points = deque(
        [(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(4)],
        maxlen=50
    )

    log_queue.put("[Process] 마우스 경로 생성 시작")
    i = 0

    click_listener = start_mouse_click_listener(stop_event)

    while not stop_event.is_set():

        # 이동 완료되면 좌표 새걸로 변경
        last_point = points[-1]
        new_point = (
            max(0, min(screen_width, last_point[0]+random.randint(-200, 200))),
            max(0, min(screen_height, last_point[1]+random.randint(-200, 200)))
        )
        points.append(new_point)
                                        
        p0, p1, p2, p3 = list(points)[-4:]
        steps = random.randint(30, 60)

        pattern = random.choices(
            ['ease', 'linear', 's_curve'],
            weights=[0.35, 0.25, 0.2]
        )[0]

        for i in range(steps):
            if stop_event.is_set():
                break
            step_start = time.time()

            t = i / steps

            # ===== 이동 패턴 계산 =====
            if pattern == 'ease':
                t_mod = ease_in_out_quad_random(t)
            elif pattern == 'linear':
                t_mod = linear(t)
            elif pattern == 'zigzag':
                t_mod = ease_in_out_quad_random(t) + math.sin(t * math.pi * 3) * 0.02
            elif pattern == 's_curve':
                t_mod = ease_in_out_s_curve(t)

            x, y = catmull_rom_spline(p0, p1, p2, p3, t_mod)

            # n% 확률로 jerk, jitter
            if random.random() < 0.05:
                x += random.randint(-1, 1)
                y += random.randint(-1, 1)
                x = max(0, min(screen_width, x))
                y = max(0, min(screen_height, y))

            if globals.Recorder == "postgres":
                record_timestamp = datetime.now()
            elif globals.Recorder == "json":
                record_timestamp = datetime.now().isoformat()   

            globals.MOUSE_QUEUE.put({
                'timestamp': record_timestamp,
                'x': int(x),
                'y': int(y),
                'event_type': 0,
                'is_pressed': globals.IS_PRESSED
            })

            if move_mouse:
                pyautogui.moveTo(int(x), int(y), duration=0.3)

            # n% 확률로 클릭
            if random.random() < 0.05 and globals.IS_PRESSED == 0:
                macro_click(int(x), int(y), move_mouse)
                
            step_end = time.time()  # 스텝 끝 시각
            print(f"[Step {i+1}/{steps}] 소요 시간: {step_end - step_start:.6f}초")
            t += random.uniform(0.015, 0.025)
        ##
        if globals.MOUSE_QUEUE.qsize() >= globals.MAX_QUEUE_SIZE:
            log_queue.put(f"Data 5000개 초과.. 누적 {5000 * i}")
            i += 1           
            # isUser => False
            cunsume_q(record=record, isUser=False, log_queue=log_queue)
            log_queue.put("저장 완료 다음 시퀀스 준비")

        time.sleep(interval)

    click_listener.stop()

    time.sleep(0.05)

    # 남아있는거 PUSH 후 종료
    cunsume_q(record=record, isUser=False, log_queue=log_queue)

    stop_event.set()
