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

def ease_in_out_quad(t): return 2*t*t if t < 0.5 else -1 + (4-2*t)*t
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
    if move_mouse:
        pyautogui.mouseDown(x=x, y=y)
    else:
        globals.IS_PRESSED = 1
        globals.MOUSE_QUEUE.put({
            'timestamp': datetime.now(),
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
            'timestamp': datetime.now(),
            'x': int(x),
            'y': int(y),
            'event_type': 2,
            'is_pressed': 0
        })

def record_mouse_path(stop_event = None, move_mouse=False, user_macro=False, record=True, interval=0.003, log_queue:Queue=None):
    if stop_event is None:
        stop_event = Event()

    # setting
    globals.IS_PRESSED = 0

    points = deque(
        [(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(4)],
        maxlen=50
    )

    log_queue.put("[Process] 마우스 경로 생성 시작")

    click_listener = start_mouse_click_listener(stop_event)

    if not user_macro:
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
                ['ease', 'linear', 'zigzag', 's_curve', 'jitter'],
                weights=[0.35, 0.15, 0.15, 0.15, 0.2]
            )[0]

            for i in range(steps):
                if stop_event.is_set():
                    break

                t = i / steps

                # ===== 이동 패턴 계산 =====
                if pattern == 'ease':
                    t_mod = ease_in_out_quad(t)
                elif pattern == 'linear':
                    t_mod = linear(t)
                elif pattern == 'zigzag':
                    t_mod = ease_in_out_quad(t) + math.sin(t * math.pi * 3) * 0.02
                elif pattern == 's_curve':
                    t_mod = ease_in_out_s_curve(t)
                else:
                    t_mod = ease_in_out_quad(t) + random.uniform(-0.03, 0.03)

                x, y = catmull_rom_spline(p0, p1, p2, p3, t_mod)

                # 극단값 진짜 필요한가?
                # # ===== 순간이동 랜덤 적용 =====
                # if random.random() < 0.02:  # 2% 확률로 순간이동
                #     x += random.randint(-400, 400)  # 순간이동 범위 조정
                #     y += random.randint(-400, 400)

                # 작은 흔들기 jerk
                x += random.randint(-3, 3)
                y += random.randint(-3, 3)
                x = max(0, min(screen_width, x))
                y = max(0, min(screen_height, y))

                globals.MOUSE_QUEUE.put({
                    'timestamp': datetime.now(),
                    'x': int(x),
                    'y': int(y),
                    'event_type': 0,
                    'is_pressed': globals.IS_PRESSED
                })

                if move_mouse:
                    pyautogui.moveTo(int(x), int(y), duration=0)

                # 1% 확률로 클릭
                if random.random() < 0.01 and globals.IS_PRESSED == 0:
                    macro_click(int(x), int(y), move_mouse)

                time.sleep(interval)

            if globals.MOUSE_QUEUE.qsize() >= globals.MAX_QUEUE_SIZE:
                log_queue.put("Data 5000개 초과.. 저장 중..")                
                # isUser => False
                cunsume_q(record=record, isUser=False, log_queue=log_queue)
                log_queue.put("저장 완료 다음 시퀀스 준비")

            time.sleep(interval)

    else:
        # 실제 사용자 마우스 기록 모드
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
                # isUser => False
                cunsume_q(record=record, isUser=False, log_queue=log_queue)
                log_queue.put("저장 완료 다음 시퀀스 준비")

            time.sleep(interval)

    click_listener.stop()

    time.sleep(0.05)

    # 남아있는거 PUSH 후 종료
    cunsume_q(record=record, isUser=False, log_queue=log_queue)

    stop_event.set()
