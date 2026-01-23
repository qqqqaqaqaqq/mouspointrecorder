import pyautogui
import time
from datetime import datetime

from app.services.macro_dectector import MacroDetector


def main(stop_event):
    detector = MacroDetector(
        model_path="app/models/weights/mouse_macro_lstm_best.pt",
        seq_len=15,
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
                print(f"🚨 MACRO | prob={result['prob']:.3f}")
            else:
                print(f"🙂 HUMAN | prob={result['prob']:.3f}")

        time.sleep(0.01)
