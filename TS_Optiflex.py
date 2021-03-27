import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import pyfiglet


"""
Tabu Search algorithm for solving complex job shop scheduling problem.
As a first step, only the basic model is considered as described in the specification word sheet.
"""


class JobShop:
    """
    Inputs: Model parameter as csv file
    """

    big_b = 9999999

    def __init__(
        self,
        maintenance_max=None,
        processing_time=None,
        maintenance_duration=None,
        n_m=None,
        machine_feasibility=None,
        buffer_capacity=None,
    ):
        self.maintenance_max = maintenance_max
        self.processing_time = processing_time
        self.maintenance_duration = maintenance_duration
        self.n_m = n_m
        self.machine_feasibility = machine_feasibility
        self.buffer_capacity = buffer_capacity

        self.individuals = []
        self.starting_time_ij = None
        self.completion_time_ij = None
        self.completion_time_irm = None
        self.starting_time_irm = None
        self.shape = None
        self.i = None
        self.j = None
        self.r = None
        self.m = None
        self.y_jirm = None
        self.iteration_number = 0
        self.p_ij = None

        self.start_time = None

    def main_run(self):
        self.start_time = time.time()
        np.random.seed(1)
        header = pyfiglet.figlet_format("OPTIFLEX", font="big")
        print(header)

        self.generate_initial_solution()
        makespan = self.calculate_fitness_function()
        print(makespan)
        while not self.determine_termination_criterion():
            self.generate_new_solution()
            self.calculate_fitness_function()

        self.plot_gantt_chart()

        time_for_calculation = (time.time() - self.start_time) / 60
        print("Total time for calculation: {:.4f} minutes".format(time_for_calculation))

    def generate_initial_solution(self):
        # Estimating the length of the used matrix dimensions
        self._estimating_initial_domains()
        priority_list = self._generate_priority_list()
        self._initial_scheduling(priority_list)

        """
        # Depending on the processing time, a new array y_jim is generated holding decision information about which
        # machine is chosen for processing of a certain operation
        self.p_ij = np.zeros((self.i, self.j))
        self.y_jirm = np.zeros(self.shape)  # np.zeros(np.shape(self.processing_time))
        weights = []
        position = [0] * self.m
        for i in range(self.i):
            for j in range(self.j):
                for m in range(self.m):
                    weights.append(self.processing_time[j, i, m])
                weights = weights / np.sum(weights)
                indices = np.arange(self.m)
                machine_index = np.random.choice(indices, p=weights)
                position_index = position[machine_index]
                self.y_jirm[j, i, position_index, machine_index] = 1
                self.p_ij[i, j] = self.processing_time[j, i, machine_index]
                # Updating helper variables
                position[machine_index] += 1
                weights = []
        """
        # Do a feasibility check
        self._check_feasibility()

        # In order to receive an initial solution, y_jim has to extended to a fourth dimension, containing the position
        # for each operation on a machine

    def calculate_fitness_function(self):
        # Initialize helper variables with its correct dimensions
        self.starting_time_irm = np.zeros((self.i, self.r + 1, self.m))
        self.starting_time_irm[0, 0, :] = 0
        self.completion_time_irm = np.zeros((self.i, self.r, self.m))
        self.starting_time_ij = np.zeros((self.i, self.j))
        self.completion_time_ij = np.zeros((self.i, self.j))

        processing_time = np.repeat(
            self.processing_time[:, :, :, np.newaxis], self.r, axis=3
        )
        processing_time = np.swapaxes(processing_time, 2, 3)

        self._equation_1(processing_time)
        self._equation_2()
        self._equation_5_and_6()
        self._equation_7()

        makespan = np.max(self.completion_time_ij)
        return makespan

    def _equation_1(self, processing_time):
        for m in range(self.m):
            addition = 0
            for r in range(self.r):
                for i in range(self.i):
                    for j in range(self.j):
                        addition += (
                            processing_time[j, i, r, m] * self.y_jirm[j, i, r, m]
                        )
                    self.starting_time_irm[i, r + 1, m] = (
                            self.starting_time_irm[i, r, m]
                            + addition)

                    """  
                    if addition != 0:
                        try:
                            self.starting_time_irm[i, r + 1, m] = (
                                self.starting_time_irm[i, r, m]
                                + addition
                                + self.starting_time_irm[i - 1, r, m]
                            )
                        except:
                            self.starting_time_irm[i, r + 1, m] = (
                                self.starting_time_irm[i, r, m] + addition
                            )
                    """

    def _equation_2(self):
        for r in range(self.r):
            self.completion_time_irm[:, r, :] = self.starting_time_irm[:, r + 1, :]

    def _equation_5_and_6(self):
        for j in range(self.j):
            for i in range(self.i):
                for r in range(self.r):
                    for m in range(self.m):
                        if self.y_jirm[j, i, r, m] == 1:
                            self.starting_time_ij[i, j] = self.starting_time_irm[
                                i, r, m
                            ]
                            self.completion_time_ij[i, j] = self.completion_time_irm[
                                i, r, m
                            ]

    def _equation_7(self):
        for j in range(self.j):
            for i in range(self.i - 1):
                if self.starting_time_ij[i + 1, j] < self.completion_time_ij[i, j]:
                    self.starting_time_ij[i + 1, j] = self.completion_time_ij[i, j]

    def determine_termination_criterion(self):
        maximum_iter = 10
        self.iteration_number += 1
        if maximum_iter <= self.iteration_number:
            return True
        return False

    def generate_new_solution(self):
        pass

    def _selecting_move_type(self):
        pass

    def _apply_move_type(self):
        pass

    def _move_operation_i(self):
        pass

    def _move_operation_ii(self):
        pass

    def _move_operation_iii(self):
        pass

    def _move_maintenance_i(self):
        pass

    def _move_maintenance_ii(self):
        pass

    def _move_buffer_i(self):
        pass

    def _move_buffer_ii(self):
        pass

    def _check_feasibility(self):
        # First feasibility check
        # Returns True if an operation of a job is only assigned to a machine where this operation is possible.
        # Word sheet equation 3: \sum_r^J y_jirm <= f_jim
        machine_feasibility = np.array(self.processing_time, dtype=bool)
        y_ijm = np.sum(self.y_jirm, axis=2)
        it = np.nditer(y_ijm, flags=["multi_index"])
        for x in it:
            x = int(x)
            y = int(machine_feasibility[it.multi_index])
            if x > y:
                print("First check failed")
                return False
        print("First check passed")

        # Second feasibility check
        # Returns True if each operation of a job is assigned to exactly one machine
        # Word sheet equation 4: \sum_r^J\sum_m^M (y_jirm * f_jim) = 1
        tensor_contraction = np.zeros(self.shape)
        it = np.nditer(self.y_jirm, flags=["multi_index"])
        for x in it:
            x = int(x)
            y = int(
                machine_feasibility[
                    it.multi_index[0], it.multi_index[1], it.multi_index[3]
                ]
            )
            tensor_contraction[it.multi_index] = x * y

        tensor_contraction = np.sum(tensor_contraction, axis=2)
        tensor_contraction = np.sum(tensor_contraction, axis=2)

        if tensor_contraction.all:
            print("Second check passed")
        else:
            print("Second check failed")
            return False

        return True

    def _estimating_initial_domains(self):
        initial_length = np.sum(self.processing_time)
        self.i = np.shape(self.processing_time)[1]
        self.j = np.shape(self.processing_time)[0]
        self.m = np.shape(self.processing_time)[2]
        self.r = int(
            np.max(
                np.sum(
                    np.sum(np.array(self.processing_time, dtype=bool), axis=0), axis=0
                )
            )
        )
        self.shape = [self.j, self.i, self.r, self.m]
        return initial_length

    def _generate_priority_list(self):
        self.p_ij = np.zeros((self.i, self.j))
        weights = []
        for i in range(self.i):
            for j in range(self.j):
                for m in range(self.m):
                    weights.append(self.processing_time[j, i, m])
                weights = weights / np.sum(weights)
                indices = np.arange(self.m)
                machine_index = np.random.choice(indices, p=weights)
                self.p_ij[i, j] = self.processing_time[j, i, machine_index]
                weights = []

        total_time = np.sum(self.p_ij, axis=0)
        job_index = np.arange(self.j)
        sort = np.argsort(total_time)
        priority_list = np.column_stack((job_index[sort], total_time[sort]))

        return priority_list

    def _initial_scheduling(self, priority_list):
        self.y_jirm = np.zeros(self.shape)
        machine_position = np.array([0] * self.m)
        for job_index in priority_list[:, 0]:
            job_index = int(job_index)
            for i in range(self.i):
                machine_index = []
                for m in range(self.m):
                    if any(np.ndarray(self.processing_time[job_index, i, m], dtype=bool)) is True:
                        machine_index.append(m)
                # get the minimum of machine position
                machine_choice = machine_index[np.argmin(machine_position[machine_index])]
                self.y_jirm[job_index, i, machine_position[machine_choice], machine_choice] = 1
                machine_position[machine_choice] += 1

    def plot_gantt_chart(self):
        import plotly.figure_factory as ff
        import plotly.io as pio
        pio.renderers.default = 'browser'
        df = []
        color = 0
        cw = lambda: np.random.randint(0, 255)
        colors = ['#%02X%02X%02X' % (cw(), cw(), cw())]

        for j in range(self.j):
            for m in range(self.m):
                for i in range(self.i):
                    if any(np.array(self.y_jirm[j, i, :, m], dtype=bool)) is True:
                        entry = dict(
                            Task=str(m),
                            Start=str(self.starting_time_ij[i, j]),
                            Finish=str(self.completion_time_ij[i, j]),
                            Resource="job " + str(j),
                        )
                        df.append(entry)

            colors.append('#%02X%02X%02X' % (cw(), cw(), cw()))
            #color += 100/self.j

        fig = ff.create_gantt(
            df, colors=colors, index_col="Resource", show_colorbar=True, group_tasks=False
        )
        fig.show()


def extract_csv(file_name, dimension=None):

    if dimension in [None, 0, 1, 2]:
        array = pd.read_csv(file_name, sep=";", index_col=0, header=0)
        return array.values
    elif dimension == 3:
        read_in = pd.read_csv(file_name, sep=";", index_col=None, header=None)
        machine_list = read_in.index[read_in[0].str.contains("m")].tolist()
        array = read_in.values[
            1 : int(read_in.shape[0] / len(machine_list)), 1 : read_in.shape[1]
        ]
        for val in machine_list:
            array = np.dstack(
                (
                    np.atleast_3d(array),
                    read_in.values[
                        val + 1 : int(read_in.shape[0] / len(machine_list) + val),
                        1 : read_in.shape[1],
                    ],
                )
            )
        return array[:, :, 1::].astype(np.int64)
    else:
        raise ValueError("Select a feasible dimension parameter")


if __name__ == "__main__":

    processing_time_path = (
        "/Users/q517174/PycharmProjects/pythonProject/input_data/processing_time.csv"
    )
    processing_time_input = extract_csv(processing_time_path, 3)

    # After read in everything the object can be created and the main_run can start
    job_shop_object = JobShop(
        processing_time=processing_time_input,
    )
    job_shop_object.main_run()
