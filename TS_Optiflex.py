import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import pyfiglet
from alive_progress import alive_bar


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
        self.completion_time_rm = None
        self.starting_time_rm = None
        self.shape = None
        self.i = None
        self.j = None
        self.r = None
        self.m = None
        self.y_jirm = None
        self.iteration_number = 0
        self.p_ij = None
        self.priority_list = None
        self.max_iter = 1

        self.feasibility = None
        self.start_time = None

    def main_run(self):
        self.start_time = time.time()
        np.random.seed(1)
        header = pyfiglet.figlet_format("OPTIFLEX", font="big")
        print(header)

        self.generate_initial_solution()
        makespan = self.calculate_fitness_function()
        print("Makespan of initial solution: {}".format(makespan))
        self.plot_gantt_chart(group=True, title="Initial Scheduling, Feasibility: {}".format(self.feasibility))

        with alive_bar(self.max_iter, title="Iterations") as bar:
            while not self.determine_termination_criterion():
                self.generate_new_solution()
                # self.calculate_fitness_function()
                time.sleep(0.5)
                bar()

        # self.plot_gantt_chart(group=True)

        time_for_calculation = (time.time() - self.start_time) / 60
        print("Total time for calculation: {:.4f} minutes".format(time_for_calculation))

    def generate_initial_solution(self):
        # Estimating the length of the used matrix dimensions
        self._estimating_initial_domains()
        self.priority_list = self._generate_priority_list()
        self._initial_scheduling(self.priority_list)

        # Do a feasibility check
        self._check_feasibility()

    def calculate_fitness_function(self):
        # Initialize helper variables with its correct dimensions
        self.starting_time_rm = np.zeros((self.r, self.m))
        self.starting_time_rm[0, :] = 0
        self.completion_time_rm = np.zeros((self.r, self.m))
        self.starting_time_ij = np.zeros((self.i, self.j))
        self.completion_time_ij = np.zeros((self.i, self.j))

        self._equation_1()
        self._equation_5_and_6()
        self.feasibility = self._scheduling()

        makespan = np.max(self.completion_time_ij)
        return makespan

    def _equation_1(self, update=False):
        # TODO: Circumvent bare exception cause
        processing_time = np.repeat(
            self.processing_time[:, :, :, np.newaxis], self.r, axis=3
        )
        processing_time = np.swapaxes(processing_time, 2, 3)
        for m in range(self.m):
            for r in range(self.r):
                addition = 0
                for i in range(self.i):
                    for j in range(self.j):
                        addition += (
                            processing_time[j, i, r, m] * self.y_jirm[j, i, r, m]
                        )
                try:
                    self.starting_time_rm[r + 1, m] = (
                        self.starting_time_rm[r, m] + addition
                    )
                except:
                    pass
                self.completion_time_rm[r, m] = self.starting_time_rm[r, m] + addition

    def _equation_5_and_6(self):
        # Transformation from rm to ij over decision variable y
        for j in range(self.j):
            for i in range(self.i):
                for r in range(self.r):
                    for m in range(self.m):
                        if self.y_jirm[j, i, r, m] == 1:
                            self.starting_time_ij[i, j] = self.starting_time_rm[r, m]
                            self.completion_time_ij[i, j] = self.completion_time_rm[
                                r, m
                            ]
        """
    def _equation_7(self):
        # TODO: Find a way to circumvent bare exception cause.
        store = []
        for r in range(self.r):
            for m in range(self.m):
                #  get i and j
                j = np.where(self.y_jirm[:, :, r, m] == 1)[0]
                i = np.where(self.y_jirm[:, :, r, m] == 1)[1]
                max_pos = np.max((np.where(self.y_jirm[:, :, :, m] == 1))[2]) + 1
                try:
                    j = int(j)
                    i = int(i)
                except:
                    break
                if i < self.i - 1:
                    if self.starting_time_ij[i + 1, j] < self.completion_time_ij[i, j]:
                        difference = (
                            self.completion_time_ij[i, j]
                            - self.starting_time_ij[i + 1, j]
                        )

                        for ii in range(i + 1, self.i):
                            r_new = int(np.where(self.y_jirm[j, ii, :, :] == 1)[0])
                            m_new = int(np.where(self.y_jirm[j, ii, :, :] == 1)[1])
                            for rr in range(r_new, max_pos):
                                # try:
                                j_new = int(
                                    np.where(self.y_jirm[:, :, rr, m_new] == 1)[0]
                                )
                                i_new = int(
                                    np.where(self.y_jirm[:, :, rr, m_new] == 1)[1]
                                )
                                # TODO: difference must be reduced by idle time between two
                                # idle time zur vorherigen position
                                if rr == r_new:
                                    self.starting_time_ij[i_new, j_new] += difference
                                    self.completion_time_ij[i_new, j_new] += difference
                                else:
                                    if (
                                        self.starting_time_ij[i_new, j_new]
                                        < self.completion_time_ij[store[1], store[0]]
                                    ):
                                        diff = self.completion_time_ij[store[1], store[0]] - self.starting_time_ij[
                                            i_new, j_new
                                        ]
                                        self.starting_time_ij[i_new, j_new] += diff
                                        self.completion_time_ij[i_new, j_new] += diff
                                store = []
                                store.append(j_new)
                                store.append(i_new)
        """

    def _scheduling(self):
        # TODO: check number of maximum iterations
        # TODO: Maybe breaking condition if both scheduling processes have alternating behavior
        max_iter = self.i * self.j
        for n in range(max_iter):
            if n == max_iter - 1:
                return False
            rm = self._schedule_rm()
            ij = self._schedule_ij()
            if rm is True & ij is True:
                return True

    def _schedule_rm(self):
        change = 0
        for m in range(self.m):
            max_pos = np.max((np.where(self.y_jirm[:, :, :, m] == 1))[2])
            for r in range(max_pos):
                j1 = np.where(self.y_jirm[:, :, r, m] == 1)[0]
                i1 = np.where(self.y_jirm[:, :, r, m] == 1)[1]
                j2 = np.where(self.y_jirm[:, :, r + 1, m] == 1)[0]
                i2 = np.where(self.y_jirm[:, :, r + 1, m] == 1)[1]
                if self.starting_time_ij[i2, j2] < self.completion_time_ij[i1, j1]:
                    difference = self.completion_time_ij[i1, j1] - self.starting_time_ij[i2, j2]
                    self.starting_time_ij[i2, j2] += difference
                    self.completion_time_ij[i2, j2] += difference
                    change += 1
        if change == 0:
            return True
        else:
            return False

    def _schedule_ij(self):
        change = 0
        for j in range(self.j):
            for i in range(self.i - 1):
                if self.starting_time_ij[i + 1, j] < self.completion_time_ij[i, j]:
                    difference = self.completion_time_ij[i, j] - self.starting_time_ij[i + 1, j]
                    self.starting_time_ij[i + 1, j] += difference
                    self.completion_time_ij[i + 1, j] += difference
                    change += 1
        if change == 0:
            return True
        else:
            return False

    def determine_termination_criterion(self):
        self.iteration_number += 1
        if self.max_iter <= self.iteration_number - 1:
            return True
        return False

    def generate_new_solution(self):
        self._selecting_move_type()
        self._apply_move_type()

    def _selecting_move_type(self):
        if self.iteration_number % 10 == 0:
            self._move_operation_insert_on_another_machine()
        elif self.iteration_number % 3 == 0:
            self._move_operation_insert_operation_on_one_machine()
        else:
            self._move_operation_position_swap_on_one_machine()

    def _apply_move_type(self):
        pass

    def _move_operation_insert_on_another_machine(self):
        pass

    def _move_operation_position_swap_on_one_machine(self):
        initial_y_jirm = copy.deepcopy(self.y_jirm)
        # Change y_jirm and calculate all other helper variables again and determine makespan
        # select the machine to perform this operation
        m = np.random.randint(0, self.m)
        # select a position on this machine
        max_pos = np.max((np.where(self.y_jirm[:, :, :, m] == 1))[2]) + 1
        r_swap = np.random.randint(0, max_pos)
        # swap this position with all others, this gives the neighborhood
        for r in range(max_pos):
            if r != r_swap:  # TODO: Additionally, check if move is in tabu list
                # reset self.y_jirm
                self.y_jirm = copy.deepcopy(initial_y_jirm)
                # swap self.y_jirm[:,:,r,m] with self.y_jirm[:,:,r_swap,m]
                swap = np.copy(self.y_jirm[:, :, r_swap, m])
                self.y_jirm[:, :, r_swap, m] = self.y_jirm[:, :, r, m]
                self.y_jirm[:, :, r, m] = swap

                makespan = self.calculate_fitness_function()
                # Some swaps are not feasible due to deadlock effects. Thus, invalid solutions occur
                print("Makespan of new neighbor: {}".format(makespan))
                self.plot_gantt_chart(
                    group=True,
                    title="Swap {} and {} on machine {}, Feasibility: {}".format(r_swap, r, m, self.feasibility),
                )
        # TODO: Selection of the best solution, y_jirm = best solution
        # TODO: Add move type to the tabu list

    def _move_operation_insert_operation_on_one_machine(self):
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
        """
        Feasibility check of the generated decision variable y_jirm with respect to two constraints:
        Feasibility check 1: Operation assignment is only possible if an operation on a machine is feasible.
        Feasibility check 2: Every operation hast to be assigned to a machine, where the operation is feasible.

        Args:
            self.y_jirm (array): Decision variable on which feasibility check is performed
            self.processing_time (array): Transformed to boolean to receive machine feasibility

        Returns:
            True: If both checks passed
            False: If at least one check fails
        """

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
            pass
        else:
            print("Second check failed")
            return False

        return True

    def _estimating_initial_domains(self):
        """
        Domains for operations, jobs, positions and machines are estimated based on the parameter processing time.
        The shape for the decision variable is given.

        Args:
            self.processing_time (array): Processing time for operation i of job j on machine m

        Returns:
            initial_length (int): Upper bound for total amount of periods
        """

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
        """
        All jobs considered in the optimization problem are assigned to a priority list, sorting the jobs
        from shortest to longest overall completion time. For each pair [i,j], only one feasible machine has to be
        selected by assigning a weight on each machine depending on its processing time for this operation. Afterwards,
        the machine is randomly selected.
        The second part of the function sums up the total processing time for each pair [i,j] and sorting this list
        with respect to the total processing time, leading to a priority list of jobs.

        Args:
            self.p_ij (array): Processing time of operation i of job j
            self.i (int): Number of operation
            self.j (int): Number of jobs
            self.m (int): Number of machines
            self.processing_time (array): Processing time of operation i of job j on machine m

        Returns:
            priority_list (array): First column containing job index and second column the total completion time
        """
        # TODO: remove self.p_ij from self, parameter only needed in this function
        self.p_ij = np.zeros((self.i, self.j))
        weights = []  # initialize weights list
        # loop over all indices of processing time
        for i in range(self.i):
            for j in range(self.j):
                for m in range(self.m):
                    weights.append(self.processing_time[j, i, m])
                weights = weights / np.sum(
                    weights
                )  # normalize weights, that sum(weights) = 1
                indices = np.arange(self.m)  # machine indices from 0 to self.m
                machine_index = np.random.choice(
                    indices, p=weights
                )  # randomly select a machine, w.r.t. weights
                self.p_ij[i, j] = self.processing_time[
                    j, i, machine_index
                ]  # store processing time in reduced array
                weights = []  # reset weights
        #  --------------------------------------------------------------------
        total_time = np.sum(
            self.p_ij, axis=0
        )  # calculate total completion time for each job (sum over i)
        job_index = np.arange(self.j)  # job indices from 0 to self.j
        sort = np.argsort(
            total_time
        )  # get indices of sorted processing time (no explicit tiebreaker considered)
        priority_list = np.column_stack(
            (job_index[sort], total_time[sort])
        )  # generate priority list

        return priority_list

    def _initial_scheduling(self, priority_list):
        """
        Depending on the priority list, the jobs will be scheduled on positions of the feasible machines. Operations
        are assigned to the machine with the lowest position index. If to feasible machines has the same position
        index, the operation is assigned to the machine with the lowest machine index. Conclusively, the job with the
        highest priority is assigned to position zero on each machine.

        Args:
            self.y_jirm (array): Processing time of operation i of job j
            self.i (int): Number of operation
            priority_list (array): Jobs sorted depending on total processing time
            self.m (int): Number of machines
            self.processing_time (array): Processing time of operation i of job j on machine m

        Returns:

        """
        self.y_jirm = np.zeros(self.shape)  # initialize decision variable
        machine_position = np.array(
            [0] * self.m
        )  # list to store used machine positions
        for job_index in priority_list[:, 0]:  # loop over priority list
            job_index = int(job_index)
            for i in range(self.i):  # loop over operations
                machine_index = []  # reset machine index
                for m in range(self.m):  # loop over machine
                    if (  # check if pair [i,j] is feasible on this machine
                        any(
                            np.ndarray(
                                self.processing_time[job_index, i, m], dtype=bool
                            )
                        )
                        is True
                    ):
                        machine_index.append(
                            m
                        )  # if feasible, assign machine to possible machines
                machine_choice = machine_index[
                    np.argmin(
                        machine_position[machine_index]
                    )  # get the minimum of machine position and choose this machine
                ]
                # set decision variable to one for [i,j] on the determined position and machine
                self.y_jirm[
                    job_index, i, machine_position[machine_choice], machine_choice
                ] = 1
                machine_position[machine_choice] += 1

    def plot_gantt_chart(self, group=False, title="Gantt_Chart"):
        """
        Plot Gantt Chart depending on the starting and completion times of all operations of each job

        Args:
            self.starting_time_ij (array):
            self.completion_time_ij (array):
            group (bool): False if machines in gantt chart are not supposed to group up, True if they are supposed
            to be grouped
            title (str): Title of plot

        Returns:

        """

        import plotly.figure_factory as ff
        import plotly.io as pio

        pio.renderers.default = "browser"
        df = []
        color = 0
        cw = lambda: np.random.randint(0, 255)
        colors = ["#%02X%02X%02X" % (cw(), cw(), cw())]

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

            colors.append("#%02X%02X%02X" % (cw(), cw(), cw()))
            # color += 100/self.j

        fig = ff.create_gantt(
            df,
            colors=colors,
            index_col="Resource",
            show_colorbar=True,
            group_tasks=group,
            title=title,
        )
        fig.update_traces(
            mode="lines", line_color="black", selector=dict(fill="toself")
        )
        # for trace in fig.data:
        #    trace.x += (trace.x[0], trace.x[0], trace.x[0])
        #    trace.y += (trace.y[-3], trace.y[0], trace.y[0])
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

    processing_time_path = "/Users/q517174/PycharmProjects/Optiflex/processing_time.csv"
    processing_time_input = extract_csv(processing_time_path, 3)

    # After read in everything the object can be created and the main_run can start
    job_shop_object = JobShop(
        processing_time=processing_time_input,
    )
    job_shop_object.main_run()
