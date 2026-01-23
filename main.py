# main.py
from app.ui.main_ui import MouseMacroUI
from app.db.session import init_db

if __name__ == "__main__":
    init_db()

    app = MouseMacroUI()
    app.mainloop()
