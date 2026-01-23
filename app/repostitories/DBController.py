# sqlalchemy
# 앱 내부
from app.db.session import SessionLocal

from app.models.MousePoint import MousePoint
import app.core.globals as globals

def point_insert():
    db = SessionLocal()

    while True:
        data = globals.MOUSE_QUEUE.get()
        if data is None:
            break

        try:
            mouse_point = MousePoint(
                timestamp=data['time'],
                x=data['x'],
                y=data['y']
            )
            db.add(mouse_point)
            db.commit()
        except Exception as e:
            db.rollback()
            print("DB 저장 오류:", e)

def point_clear():
    db = SessionLocal()
    try:
        db.query(MousePoint).delete()
        db.commit()

        db.execute("DELETE FROM sqlite_sequence WHERE name='MousePoint';")
        db.commit()

        print("MousePoint 테이블 초기화 완료")
    except Exception as e:
        db.rollback()
        print("테이블 초기화 오류:", e)
    finally:
        db.close()

def read():
    db = SessionLocal()
    try:
        point = db.query(MousePoint).all()
        return point
    except Exception as e:
        print("DB 읽기 오류:", e)
        return []
    finally:
        db.close()