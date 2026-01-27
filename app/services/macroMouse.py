import os
import json

from multiprocessing import Event
import pyautogui
import time, random, math
from datetime import datetime

import app.core.globals as globals

pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

def catmull_rom_spline(p0, p1, p2, p3, t):
    t2 = t*t
    t3 = t2*t
    x = 0.5*((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
    y = 0.5*((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
    return x, y

def ease_in_out_quad(t): return 2*t*t if t<0.5 else -1 + (4-2*t)*t
def linear(t): return t
def ease_out_cubic(t): return 1 - pow(1-t, 3)

def record_mouse_path(stop_event: Event = None, move_mouse=False, user_macro=False, record=True, segments=5, interval = 0.01):
    if stop_event is None:
        stop_event = Event()

    points = [(random.randint(0, screen_width), random.randint(0, screen_height)) for _ in range(4)]
    all_data = [] 

    print("[Process] 마우스 경로 생성 시작")

    if user_macro:
        while not stop_event.is_set():
            for _ in range(segments):
                p0,p1,p2,p3 = points[-4:]
                steps = random.randint(30,60)
                pattern = random.choices(['ease','linear','zigzag','pause'], weights=[0.5,0.2,0.2,0.1])[0]

                for i in range(steps):
                    if stop_event.is_set():
                        break
                    
                    t = i / steps
                    t_mod = ease_in_out_quad(t) if pattern=='ease' else \
                            linear(t) if pattern=='linear' else \
                            ease_in_out_quad(t)+math.sin(t*math.pi*random.randint(2,5))*0.02 if pattern=='zigzag' else \
                            ease_out_cubic(t)
                    x,y = catmull_rom_spline(p0,p1,p2,p3,t_mod)
                    x += random.uniform(-2,2)
                    y += random.uniform(-2,2)
                    x = max(0,min(screen_width,x))
                    y = max(0,min(screen_height,y))

                    timestamp = datetime.now()
                    data = {'timestamp': timestamp, 'x': int(x), 'y': int(y)}
                    if record: all_data.append(data)
                    if move_mouse: pyautogui.moveTo(int(x),int(y),duration=0)
                    if pattern=='pause' and random.random()<0.05: time.sleep(random.uniform(0.05,0.2))

                    time.sleep(interval)

                points.append((random.randint(0,screen_width), random.randint(0,screen_height)))
        else:
            while not stop_event.is_set():
                x, y = pyautogui.position()
                timestamp = datetime.now()
                data = {'timestamp': timestamp, 'x': x, 'y': y}

                globals.MOUSE_QUEUE.put(data)

                if record:
                    all_data.append(data)

                time.sleep(interval)
                        

        # DB 저장
        if record and globals.Recorder == "postgres":
            if all_data:
                from app.repostitories.DBController import SessionLocal, MacroMousePoint
                db = SessionLocal()
                try:
                    for item in all_data:
                        mp = MacroMousePoint(timestamp=item['timestamp'], x=item['x'], y=item['y'])
                        db.add(mp)
                    db.commit()
                    print(f"[Process] 총 {len(all_data)}개 포인트 DB 저장 완료")
                except Exception as e:
                    db.rollback()
                    print("[Process] DB 저장 오류:", e)
                finally:
                    db.close()
        elif record and globals.Recorder == "json":
            if all_data:
                # 저장 경로, 현재 디렉토리 기준
                save_dir = os.path.join(globals.JsonPath, "macro")
                os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
                # 파일 이름: 현재 시간 기준
                file_name = f"macro_move.json"
                file_path = os.path.join(save_dir, file_name)

                try:
                    # JSON으로 저장
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(all_data, f, ensure_ascii=False, indent=4, default=str)
                    print(f"[Process] 총 {len(all_data)}개 포인트 JSON 저장 완료: {file_path}")
                except Exception as e:
                    print("[Process] JSON 저장 오류:", e)

    stop_event.set()