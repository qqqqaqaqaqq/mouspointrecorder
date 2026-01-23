import threading

from app.services.macroMouse import record_mouse_path
from app.repostitories.DBController import macro_point_insert
import app.core.globals as globals

def main(stop_event=None):
    """Mouse 기록 메인 함수. stop_event가 설정되면 루프 종료"""
    
    # DB 저장 스레드 시작
    db_thread = threading.Thread(target=macro_point_insert, daemon=True)
    db_thread.start()

    print("Mouse Move Start")
    # stop_event를 record_mouse_path에 전달
    record_mouse_path(stop_event=stop_event)

    globals.MOUSE_QUEUE.put(None)
    db_thread.join()
    print("프로그램 종료")
