import matplotlib.pyplot as plt
from multiprocessing import Queue

def plot_main(points, interval=0.01, log_queue: Queue = None):
    if len(points) == 0:
        print('DB에 저장된 point가 없습니다.')
        return

    if log_queue:
        log_queue.put("plot 실행")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    # 이동 경로
    line, = ax.plot([], [], color='blue', linewidth=1, alpha=0.6, label='Move Path')
    move_scatter = ax.scatter([], [], c='blue', s=8, alpha=0.5)

    # 클릭 이벤트
    click_down_scatter = ax.scatter([], [], c='red', s=50, marker='o', label='Click Down')
    click_up_scatter = ax.scatter([], [], c='green', s=50, marker='^', label='Click Up')

    # 눌린 상태 유지 표시 (드래그)
    pressed_scatter = ax.scatter([], [], c='red', s=20, alpha=0.3, marker='o', label='Pressed')

    ax.set_title("Mouse Movement + Click / Press State")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    move_x, move_y = [], []
    down_x, down_y = [], []
    up_x, up_y = [], []
    pressed_x, pressed_y = [], []

    idx = 0

    while plt.fignum_exists(fig.number):
        if idx < len(points):
            p = points[idx]

            # dict / ORM 객체 대응
            if isinstance(p, dict):
                x = p.get("x", 0)
                y = p.get("y", 0)
                event_type = p.get("event_type", 0)
                is_pressed = p.get("is_pressed", 0)
            else:
                x = getattr(p, "x", 0)
                y = getattr(p, "y", 0)
                event_type = getattr(p, "event_type", 0)
                is_pressed = getattr(p, "is_pressed", 0)

            # 이동
            if event_type == 0:
                move_x.append(x)
                move_y.append(y)
                line.set_data(move_x, move_y)
                move_scatter.set_offsets(list(zip(move_x, move_y)))

                # 눌린 상태면 pressed marker 유지
                if is_pressed == 1:
                    pressed_x.append(x)
                    pressed_y.append(y)
                    pressed_scatter.set_offsets(list(zip(pressed_x, pressed_y)))

            # 클릭 down
            elif event_type == 1:
                down_x.append(x)
                down_y.append(y)
                click_down_scatter.set_offsets(list(zip(down_x, down_y)))

            # 클릭 up
            elif event_type == 2:
                up_x.append(x)
                up_y.append(y)
                click_up_scatter.set_offsets(list(zip(up_x, up_y)))

            idx += 1

            # 화면 범위 자동 조정
            all_x = move_x + down_x + up_x + pressed_x
            all_y = move_y + down_y + up_y + pressed_y

            if all_x and all_y:
                ax.set_xlim(0, max(all_x) + 50)
                ax.set_ylim(0, max(all_y) + 50)
                ax.invert_yaxis()

        plt.pause(interval)

    plt.ioff()

    if log_queue:
        log_queue.put("[Process] Plot 창 종료")
