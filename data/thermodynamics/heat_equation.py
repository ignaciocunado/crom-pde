import numpy as np
import os
import h5py

class HeatEquationDataGenerator:
    """
    Logic for generating data for the heat equation PDE
    """

    def __init__(self, dom_length, time, spatial_points, nt, initial_temp, nu_min, nu_max):
        self.dom_length = dom_length
        self.time = time
        self.spatial_points = spatial_points
        self.dx = dom_length / (spatial_points - 1)
        self.nt = nt
        self.dt = self.time / nt
        self.initial_temp = initial_temp
        self.grid = np.linspace(0, dom_length, spatial_points)
        self.nu_min = nu_min
        self.nu_max = nu_max


    def __generate_diffusion_coefficients(self):
        """
        Set nu to different piecewise constant functions of three regions
        :return: The spatially varying diffusion coefficient
        """

        nu1 = np.random.uniform(self.nu_min, self.nu_max)
        nu2 = np.random.uniform(self.nu_min, self.nu_max)
        nu3 = np.random.uniform(self.nu_min, self.nu_max)

        nu_x = np.zeros(self.spatial_points)
        nu_x[:self.spatial_points // 3] = nu1
        nu_x[self.spatial_points // 3:2 * self.spatial_points // 3] = nu2
        nu_x[2 * self.spatial_points // 3:] = nu3

        return nu_x

    def __solve_heat_equation(self, nu):
        """
        Solves the heat equation using finite differences
        :return the data itself
        """

        u = self.initial_temp(self.grid).copy()
        u_prev = u.copy()

        data_sequence = []

        for n in range(self.nt):
            finite_differences = (u_prev[:-2] + u_prev[2:] - 2 * u_prev[1:-1]) / self.dx ** 2
            next_time_step_velocity =  nu[1:-1] * finite_differences
            u[1:-1] = u_prev[1:-1] + self.dt * next_time_step_velocity

            # Update solution and enforce boundary conditions
            u[0], u[-1] = 0, 0

            # Swap references instead of copying
            u, u_prev = u_prev, u

            data_sequence.append(u_prev.copy())

        return np.array(data_sequence)

    def generate_data(self, n, save=False):
        """
        Generates data according to the heat equation
        :param n: number of different PDE temporal sequences
        :param save: whether to save the data in h5 format
        :return: data
        """
        parent_dir = './output'
        os.makedirs(parent_dir, exist_ok=True)
        training_data = []

        for sim_idx in range(n):
            sim_dir = os.path.join(parent_dir, str(sim_idx), f"sim_seq_{sim_idx:03d}")
            os.makedirs(sim_dir, exist_ok=True)

            nu_x_train = self.__generate_diffusion_coefficients()
            data_sequence_train = self.__solve_heat_equation(nu_x_train)
            training_data.append((np.unique(nu_x_train), data_sequence_train))

            if save:
                for time_step, state in enumerate(data_sequence_train):
                    file_name = os.path.join(sim_dir, f"h5_f_{time_step:010d}.h5")
                    with h5py.File(file_name, 'w') as h5f:
                        h5f.create_dataset('/x', data=[self.grid])
                        h5f.create_dataset('/q', data=[state])
                        h5f.create_dataset('/time', data=np.array([[time_step * self.dt]], dtype=np.float32))

                mean_x = np.mean(self.grid)
                std_x = np.std(self.grid)

                mean_q = np.mean(data_sequence_train.flatten())
                std_q = np.std(data_sequence_train.flatten())

                min_x = np.min(self.grid)
                max_x = np.max(self.grid)

                with open(f'output/{sim_idx}/meanandstd_x.npy', 'wb') as f:
                    np.save(f, mean_x)
                    np.save(f, std_x)

                with open(f'output/{sim_idx}/meanandstd_q.npy', 'wb') as f:
                    np.save(f, mean_q)
                    np.save(f, std_q)

                with open(f'output/{sim_idx}/minandmax_x.npy', 'wb') as f:
                    np.save(f, min_x)
                    np.save(f, max_x)

        return training_data
