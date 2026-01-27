import os
import json
from multiprocessing import Event
import pyautogui
import time, random, math
from datetime import datetime
import app.core.globals as globals
from multiprocessing import Queue

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

# ----------------- 마우스 경로 기록 -----------------
def record_mouse_path(stop_event: Event = None, move_mouse=False, user_macro=False, record=True, segments=5, interval=0.01, log_queue:Queue=None):
    if stop_event is None:
        stop_event = Event()

    points = [(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(4)]
    all_data = []

    if log_queue:
        log_queue.put("[Process] 마우스 경로 생성 시작")

    if not user_macro:
        while not stop_event.is_set():
            for _ in range(segments):
                p0, p1, p2, p3 = points[-4:]
                steps = random.randint(30, 60)
                pattern = random.choices(
                    ['ease', 'linear', 'zigzag', 'pause', 's_curve', 'jitter'],
                    weights=[0.35, 0.15, 0.15, 0.05, 0.15, 0.15])[0]

                for i in range(steps):
                    if stop_event.is_set(): 
                        log_queue.put("[Process] 마우스 경로 생성 종료")
                        break
                    t = i / steps

                    # 패턴별 t 변환
                    if pattern == 'ease':
                        t_mod = ease_in_out_quad(t)
                    elif pattern == 'linear':
                        t_mod = linear(t)
                    elif pattern == 'zigzag':
                        t_mod = ease_in_out_quad(t) + math.sin(t*math.pi*random.randint(2,5))*0.02
                    elif pattern == 's_curve':
                        t_mod = ease_in_out_s_curve(t)
                    elif pattern == 'jitter':
                        t_mod = ease_in_out_quad(t) + random.uniform(-0.03, 0.03)
                    else:  # pause 또는 기타
                        t_mod = ease_out_cubic(t)

                    # 좌표 계산 + 흔들림 추가
                    x, y = catmull_rom_spline(p0, p1, p2, p3, t_mod)
                    x += random.uniform(-3,3)
                    y += random.uniform(-3,3)
                    x = max(0, min(screen_width, x))
                    y = max(0, min(screen_height, y))

                    timestamp = datetime.now()
                    data = {'timestamp': timestamp, 'x': int(x), 'y': int(y)}
                    if record: all_data.append(data)
                    if move_mouse: pyautogui.moveTo(int(x), int(y), duration=0)

                    # 랜덤 pause
                    if random.random() < 0.05:  # 5% 확률
                        time.sleep(random.uniform(0.05, 0.25))

                    # 랜덤 속도 변형
                    time.sleep(interval * random.uniform(0.7, 1.3))

                # 다음 랜덤 포인트 생성
                last_point = points[-1]
                new_point = (
                    max(0, min(screen_width, last_point[0]+random.randint(-200, 200))),
                    max(0, min(screen_height, last_point[1]+random.randint(-200, 200)))
                )
                points.append(new_point)

    else:
        # 실제 사용자 마우스 기록 모드
        while not stop_event.is_set():
            x, y = pyautogui.position()
            timestamp = datetime.now()
            data = {'timestamp': timestamp, 'x': x, 'y': y}

            globals.MOUSE_QUEUE.put(data)
            if record: all_data.append(data)

            time.sleep(interval)

    # ----------------- DB 또는 JSON 저장 -----------------
    if record and globals.Recorder == "postgres":
        if all_data:
            from app.repostitories.DBController import SessionLocal, MacroMousePoint
            db = SessionLocal()
            try:
                for item in all_data:
                    mp = MacroMousePoint(timestamp=item['timestamp'], x=item['x'], y=item['y'])
                    db.add(mp)
                db.commit()
                if log_queue.put:
                    log_queue.put(f"[Process] 총 {len(all_data)}개 포인트 DB 저장 완료")
            except Exception as e:
                db.rollback()
                if log_queue.put:
                    log_queue.put(f"[Process] DB 저장 오류: {e}")
            finally:
                db.close()
    elif record and globals.Recorder == "json":
        if all_data:
            save_dir = os.path.join(globals.JsonPath, "macro")
            os.makedirs(save_dir, exist_ok=True)
            file_name = "macro_move.json"
            file_path = os.path.join(save_dir, file_name)

            try:
                # 기존 JSON 읽기
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            existing_data = json.load(f)
                            if not isinstance(existing_data, list):
                                existing_data = []
                        except json.JSONDecodeError:
                            existing_data = []
                else:
                    existing_data = []

                # 기존 데이터에 새 데이터 추가
                existing_data.extend(all_data)

                # 다시 저장
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4, default=str)

                if log_queue.put:
                    log_queue.put(f"[Process] 총 {len(all_data)}개 포인트 JSON 추가 저장 완료: {file_path}")

            except Exception as e:
                if log_queue.put:
                    log_queue.put(f"[Process] JSON 저장 오류: {e}")

    stop_event.set()
