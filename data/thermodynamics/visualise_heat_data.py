import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from heat_equation import HeatEquationDataGenerator
from initial_temperatures import *

def visualise_initial_and_final_results():
    generator = HeatEquationDataGenerator(dom_length=1.0, time=0.1, spatial_points=101, nt=2000,
                                          initial_temp=initial_temp_sinusoidal, nu_min=0.0, nu_max=0.4)
    data = generator.generate_data(4)

    x = np.linspace(0, 1, 101)

    fig, axis = plt.subplots(4, 2, figsize=(15, 20))
    axis = axis.flatten()

    i = 0
    for index, (nu_x, data_sequence) in enumerate(data):
        u_final = data_sequence[-1]
        nu_x = np.round(nu_x, 3)

        axis[i].plot(x, initial_temp_sinusoidal(x), label="Initial Condition")

        axis[i + 1].plot(np.linspace(0, 1.0, 101), u_final, label=f"Training Case {i + 1}")

        axis[i + 1].axvline(1.0 / 3, color="gray", linestyle="--", alpha=0.7)
        axis[i + 1].axvline(2 * 1.0 / 3, color="gray", linestyle="--", alpha=0.7)

        axis[i].set_xlabel("x")
        axis[i].set_ylabel("u")
        axis[i].set_title(f"Training Case {i + 1} - Initial")
        axis[i].grid()
        axis[i].legend()

        axis[i + 1].set_xlabel("x")
        axis[i + 1].set_ylabel("u")
        axis[i + 1].set_title(f"Training Case {i + 1} - Final. Nu values are: {nu_x}")
        axis[i + 1].grid()
        axis[i + 1].legend()
        i += 2

    plt.show()


def visualize_process_animation():
    generator = HeatEquationDataGenerator(dom_length=1.0, time=0.1, spatial_points=101, nt=50000,
                                          initial_temp=initial_temp_sinusoidal, nu_min=0.0, nu_max=0.2)
    data = generator.generate_data(1)

    nu_values = np.unique(data[0][0])
    data_sequence_train = data[0][1]

    fig, axis = plt.subplots(1, 1, figsize=(10, 6))

    x = np.linspace(0, 1, 101)
    line_final, = axis.plot(x, data_sequence_train[0, :], label="Final State")

    axis.set_xlabel("x")
    axis.set_ylabel("u")
    axis.set_title(f"Training Case 1 - Final. Nu values are: {np.round(nu_values, 3)}")
    axis.axvline(1.0 / 3, color="gray", linestyle="--", alpha=0.7)
    axis.axvline(2 * 1.0 / 3, color="gray", linestyle="--", alpha=0.7)
    axis.grid()
    axis.legend()

    def update(frame):
        u_final = data_sequence_train[frame, :]
        line_final.set_ydata(u_final)
        axis.set_title(f"Training Case 1 - Final. Nu values are: {np.round(nu_values, 3)} - Time Step: {frame}")
        return line_final,

    frames = range(0, int(len(data_sequence_train) / 2), 25)

    ani = FuncAnimation(fig, update, frames=frames, interval=1, blit=True)

    ani.save('animation.mp4')
    plt.show()


if __name__ == '__main__':
    visualise_initial_and_final_results()
