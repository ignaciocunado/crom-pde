import numpy as np

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

    def generate_data(self, n):
        """
        Generates data according to the heat equation
        :param n: number of samples
        :return: data
        """
        training_data = []
        for _ in range(n):  # Generate data for 8 different diffusion profiles
            nu_x_train = self.__generate_diffusion_coefficients()
            data_sequence_train = self.__solve_heat_equation(nu_x_train)
            training_data.append((np.unique(nu_x_train), data_sequence_train))
        return training_data