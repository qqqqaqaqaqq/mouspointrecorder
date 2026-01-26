import pyautogui
import time
import app.core.globals as globals
from datetime import datetime

from app.services.macro_dectector import MacroDetector

def add_macro_log(line):
    globals.MACRO_DETECTOR.append(line)
    # 리스트 최대 길이 유지
    if len(globals.MACRO_DETECTOR) > 100:
        globals.MACRO_DETECTOR.pop(0)

def main(stop_event):
    detector = MacroDetector(
        model_path="app/models/weights/mouse_macro_lstm_best.pt",
        seq_len=globals.SEQ_LEN,
        threshold=0.8
    )
    print("🟢 Macro Detector Running")

    while True:
        if stop_event.is_set():
            print("🛑 Macro Detector Stopped")
            break

        x, y = pyautogui.position()
        result = detector.push(x, y, datetime.now())

        if result:
            if result["is_macro"]:
                add_macro_log(f"🚨 MACRO | prob={result['prob']:.3f}")
            else:
                add_macro_log(f"🙂 HUMAN | prob={result['prob']:.3f}")

        time.sleep(0.01)