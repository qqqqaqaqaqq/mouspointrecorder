from app.models.MousePoint import MousePoint
import matplotlib.pyplot as plt

def plot_main(points: list[MousePoint]):
    """DB에서 받은 마우스 좌표 순서대로 PLOT 처리"""
    if len(points) == 0:
        print('DB에 저장된 point가 없습니다.')
        return

    x_values = [p.x for p in points]
    y_values = [p.y for p in points]

    x_max = max(x_values) + 100
    y_max = max(y_values) + 100

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, color='blue', linewidth=1, alpha=0.7, label='Mouse Path')
    plt.scatter(x_values, y_values, c='red', s=15, alpha=0.6, label='Points')
    plt.title(f"Mouse Movement Plot (Time Order) - Total Points: {len(points)}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()