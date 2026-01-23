from app.models.MousePoint import MousePoint
import matplotlib.pyplot as plt

def plot_main(points: list[MousePoint], interval=0.01):
    """마우스 좌표를 실시간으로 업데이트, 창 닫으면 바로 종료"""
    if len(points) == 0:
        print('DB에 저장된 point가 없습니다.')
        return

    plt.ion()  # interactive 모드 켜기
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], color='blue', linewidth=1, alpha=0.7, label='Mouse Path')
    scatter = ax.scatter([], [], c='red', s=15, alpha=0.6, label='Points')
    ax.set_title("Mouse Movement Plot (Time Order)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    x_values, y_values = [], []
    idx = 0

    while plt.fignum_exists(fig.number):  # 창이 살아있는 동안만 반복
        if idx < len(points):
            p = points[idx]
            x_values.append(p.x)
            y_values.append(p.y)
            idx += 1

            line.set_data(x_values, y_values)
            scatter.set_offsets(list(zip(x_values, y_values)))

            # x, y 축 범위 자동 조정
            ax.set_xlim(0, max(x_values) + 100)
            ax.set_ylim(0, max(y_values) + 100)
            ax.invert_yaxis()

        plt.pause(interval)  # interval 만큼 대기하면서 업데이트

    plt.ioff()  # interactive 모드 끄기
