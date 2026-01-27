import os
import json

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

    all_data = []

    print("[Process] 마우스 경로 생성 시작")
    while not stop_event.is_set():
        x, y = pyautogui.position()
        timestamp = datetime.now()
        data = {'timestamp': timestamp, 'x': x, 'y': y}

        globals.MOUSE_QUEUE.put(data)

        if record:
            all_data.append(data)

        time.sleep(interval)
        
    if record and globals.Recorder == "postgres":
        db = SessionLocal()
        try:
            for item in all_data:
                mp = MousePoint(timestamp=item['timestamp'], x=item['x'], y=item['y'])
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
            save_dir = os.path.join(globals.JsonPath, "user")
            os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
            # 파일 이름: 현재 시간 기준
            file_name = f"user_move.json"
            file_path = os.path.join(save_dir, file_name)

            try:
                # JSON으로 저장
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=4, default=str)
                print(f"[Process] 총 {len(all_data)}개 포인트 JSON 저장 완료: {file_path}")
            except Exception as e:
                print("[Process] JSON 저장 오류:", e)

    stop_event.set()