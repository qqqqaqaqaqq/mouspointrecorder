from pynput.mouse import Controller
from pynput import mouse

import time
import app.core.globals as globals
from datetime import datetime
from multiprocessing import Queue

from app.services.macro_dectector import MacroDetector
from app.core.logger import add_macro_log

def start_mouse_click_listener(stop_event):
    def on_click(x, y, button, pressed):
        if globals.Recorder == "postgres":
            record_timestamp = datetime.now()
        elif globals.Recorder == "json":
            record_timestamp = datetime.now().isoformat()      

        if stop_event.is_set():
            return False  # 리스너 종료

        event_type = 1 if pressed else 2
        globals.IS_PRESSED = 1 if pressed else 0

        data = {
            'timestamp': record_timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': event_type,
            'is_pressed': globals.IS_PRESSED
        }

        globals.MOUSE_QUEUE.put(data)

    listener = mouse.Listener(on_click=on_click)
    listener.start()
    return listener

def main(stop_event, interval=0.005, log_queue:Queue=None):
    detector = MacroDetector(
        model_path="app/models/weights/mouse_macro_lstm_best.pt",
        seq_len=globals.SEQ_LEN,
        threshold=globals.threshold
    )
    log_queue.put("🟢 Macro Detector Running")
    mouse = Controller()
    click_listener = start_mouse_click_listener(stop_event)
    
    tolerance = 0.00005
    pre_timestamp = None
    while True:
        if stop_event.is_set():
            log_queue.put("🛑 Macro Detector Stopped")
            break

        x, y = mouse.position
        timestamp = datetime.now().timestamp()
        record_timestamp = datetime.now()

        if pre_timestamp is not None:
            if (timestamp - pre_timestamp) < tolerance:
                continue 
        
        pre_timestamp = timestamp
        
        data = {
            'timestamp': record_timestamp,
            'x': int(x),
            'y': int(y),
            'event_type': 0,
            'is_pressed': globals.IS_PRESSED
        }
                
        globals.MOUSE_QUEUE.put(data)

        result = None 
        while not globals.MOUSE_QUEUE.empty():
            real_data = globals.MOUSE_QUEUE.get() 
            result = detector.push(real_data) 

        if result:
            if result["is_macro"]:
                log_queue.put(f"🚨 MACRO | prob={result['prob']:.3f}")
            else:
                log_queue.put(f"🙂 HUMAN | prob={result['prob']:.3f}")

    click_listener.stop()

    stop_event.set()    