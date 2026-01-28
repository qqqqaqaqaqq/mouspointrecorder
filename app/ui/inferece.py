import pyautogui
from pynput import mouse

import time
import app.core.globals as globals
from datetime import datetime

from app.services.macro_dectector import MacroDetector
from app.core.logger import add_macro_log

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

def main(stop_event):
    detector = MacroDetector(
        model_path="app/models/weights/mouse_macro_lstm_best.pt",
        seq_len=globals.SEQ_LEN,
        threshold=globals.threshold
    )
    add_macro_log("🟢 Macro Detector Running")

    click_listener = start_mouse_click_listener(stop_event)

    while True:
        if stop_event.is_set():
            add_macro_log("🛑 Macro Detector Stopped")
            break

        x, y = pyautogui.position()
        globals.MOUSE_QUEUE.put({
            'timestamp': datetime.now(),
            'x': int(x), 'y': int(y),
            'event_type': 0, 'is_pressed': globals.IS_PRESSED
        })

        result = None 
        while not globals.MOUSE_QUEUE.empty():
            real_data = globals.MOUSE_QUEUE.get() 
            result = detector.push(real_data) 

        if result:
            if result["is_macro"]:
                add_macro_log(f"🚨 MACRO | prob={result['prob']:.3f}")
            else:
                add_macro_log(f"🙂 HUMAN | prob={result['prob']:.3f}")

        time.sleep(0.01)

    click_listener.stop()

    stop_event.set()    