import pyautogui
import time
from datetime import datetime
import random
import math
import app.core.globals as globals

pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

def catmull_rom_spline(p0, p1, p2, p3, t):
    """Catmull-Rom 스플라인 중간점 계산"""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2*p1[0]) +
               (-p0[0] + p2[0])*t +
               (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 +
               (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
    y = 0.5 * ((2*p1[1]) +
               (-p0[1] + p2[1])*t +
               (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 +
               (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
    return x, y

def record_mouse_path(interval=0.01, stop_event=None, move_mouse=False):
    """
    진짜 매크로 느낌 좌표 생성기
    move_mouse=True 면 화면 마우스도 이동
    """
    # 랜덤 시작 위치
    points = [(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(4)]

    while True:
        if stop_event and stop_event.is_set():
            print("고급 매크로 경로 생성 중지")
            break

        # Catmull-Rom 스플라인으로 곡선 이동
        p0, p1, p2, p3 = points[-4:]
        steps = random.randint(20, 50)  # 한 구간 이동 step 수
        for i in range(steps):
            t = i / steps

            # 곡선 계산
            x, y = catmull_rom_spline(p0, p1, p2, p3, t)

            # 랜덤 흔들림 추가
            x += random.uniform(-1, 1)
            y += random.uniform(-1, 1)

            # 화면 밖 안 나가게 제한
            x = max(0, min(screen_width, x))
            y = max(0, min(screen_height, y))

            # MOUSE_QUEUE 저장
            timestamp = datetime.now()
            data = {'time': timestamp, 'x': int(x), 'y': int(y)}
            globals.MOUSE_QUEUE.put(data)

            # 실제 마우스 이동
            if move_mouse:
                pyautogui.moveTo(int(x), int(y))

            # 속도 변화: 구간마다 조금 랜덤하게 interval 조정
            time.sleep(interval * random.uniform(0.8, 1.5))

        # 다음 목표점 생성
        new_point = (random.randint(0, screen_width), random.randint(0, screen_height))
        points.append(new_point)
