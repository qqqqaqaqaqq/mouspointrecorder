import pyautogui
from multiprocessing import Event

from datetime import datetime
import keyboard
import threading
import time

import app.core.globals as globals
from app.repostitories.DBController import SessionLocal, MousePoint

pyautogui.FAILSAFE = True
screen_width, screen_height = pyautogui.size()

def record_mouse_path(stop_event: Event = None, record=True, interval=0.01):
    if stop_event is None:
        stop_event = Event()

    def wait_for_q():
        keyboard.wait('q')
        stop_event.set()
        print("[Process] Q 입력 감지: 마우스 이동 종료")
    
    threading.Thread(target=wait_for_q, daemon=True).start()

    all_data = []

    print("[Process] 마우스 경로 생성 시작")
    while not stop_event.is_set():
        x, y = pyautogui.position()
        timestamp = datetime.now()
        data = {'time': timestamp, 'x': x, 'y': y}

        globals.MOUSE_QUEUE.put(data)

        if record:
            all_data.append(data)

        time.sleep(interval)

    if record and all_data:
        
        db = SessionLocal()
        try:
            for item in all_data:
                mp = MousePoint(timestamp=item['time'], x=item['x'], y=item['y'])
                db.add(mp)
            db.commit()
            print(f"[Process] 총 {len(all_data)}개 포인트 DB 저장 완료")
        except Exception as e:
            db.rollback()
            print("[Process] DB 저장 오류:", e)
        finally:
            db.close()