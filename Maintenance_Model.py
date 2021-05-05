import pandas as pd
import numpy as np
import time
import warnings
import math
import matplotlib.pyplot as plt
import copy
import pyfiglet
from alive_progress import alive_bar

from extract_csv import extract_parameter

# TODO: Implementation of the maintenance sub-model
# TODO: Constraint neighborhood for larger data set
# TODO: Calculate idle time within job to estimate optimization potential
# TODO: Invert order of jobs in initial solution
# TODO: Test new ideas of movetypes
# TODO: Check that every processing time is greater zero
"""
Tabu Search algorithm for solving complex job shop scheduling problems.
As a first step, only the basic model is considered as described in the specification word sheet.
"""


class JobShop:
    """
    Args:
        processing_time (ndarray): Model parameter as csv file
        maintenance_max (int): Maximum number of simultaneous maintenance activities
        maintenance duration (array): Duration of maintenance activity for every task
        n_m (list): Total number of jobs on machine m
        buffer_capacity (list): Buffer capacity for each machine
        max_iter (int): Maximum number of optimization iterations
    """

    big_b = 9999999

    def __init__(
        self,
        maintenance_max=None,
        processing_time=None,
        maintenance_duration=None,
        n_m=None,
        buffer_capacity=None,
        max_iter=None,
    ):
        self.maintenance_max = maintenance_max
        self.processing_time = processing_time
        self.maintenance_duration = maintenance_duration
        self.n_m = n_m
        self.buffer_capacity = buffer_capacity
        self.max_iter = max_iter

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
        self.priority_list = None
        self.make_span = []
        self.makespan = None
        self.optimized_solution = None
        self.colors = None
        self.machine_position = None
        self.machine_time = None
        self.starting_time_mw = None
        self.completion_time_mw = None
        self.z_mu = None
        self.y_mwr = None
        self.initial_maintenance_duration = copy.deepcopy(maintenance_duration)

        self.feasibility = None
        self.start_time = None

    def main_run(self):
        """
        Main routine to perform the optimization algorithm. After generating an initial solution, new solutions
        are generated to increase the makespan. This is done while the termination criterion is not reached.
        The Gantt Chart of the initial scheduling and the optimized solution is given as well as the total
        computation time for the optimization.

        Args:

        Returns:

        """
        self.start_time = time.time()
        np.random.seed(
            20
        )  # Fix seed in order to make multiple optimization runs equivalent
        header = pyfiglet.figlet_format("OPTIFLEX", font="big")
        print(header)

        self.generate_initial_solution()
        print(self.makespan)
        self.plot_gantt_chart(
            group=True,
            title="Initial Scheduling with a makespan of {}".format(self.makespan),
        )

        self.calculate_fitness_function(0)
        makespan = []
        for j in range(self.j):
            makespan.append(np.max(self.completion_time_ij[j]))
        self.makespan = max(makespan)
        self.make_span.append(self.makespan)
        print('Initial makespan with improved scheduling order: {}'.format(self.makespan))

        with alive_bar(self.max_iter, title="Iterations") as bar:
            while not self.determine_termination_criterion():
                self.generate_new_solution()
                bar()

        if self.optimized_solution is not None:
            self.y_jirm = copy.deepcopy(self.optimized_solution[0])
            self.y_mwr = copy.deepcopy(self.optimized_solution[1])
            self.calculate_fitness_function(0)
            makespan = []
            for j in range(self.j):
                makespan.append(np.max(self.completion_time_ij[j]))
            self.makespan = max(makespan)
            self.plot_gantt_chart(
                group=True,
                title="Optimized solution with makespan of {}".format(
                    self.makespan
                ),
            )
        else:
            print(
                "\n\nNo further optimum found. Initial Solution is already optimal!\n\n"
            )

        time_for_calculation = (time.time() - self.start_time) / 60
        print("Total time for calculation: {:.4f} minutes".format(time_for_calculation))

    def generate_initial_solution(self):
        """
        The initial solution is generated after a fix scheme. First, a priority list is generated, sorting
        the jobs in order of their total completion time. Afterward, these jobs are scheduled to the machines
        on the positions according to the priority list. Finally, a feasibility check is performed in order to
        guarantee the use of a feasible initial solution.

        Args:

        Returns:

        """
        # Estimating the length of the used matrix dimensions
        self._estimating_initial_domains()
        self.priority_list = self._generate_priority_list()
        self._initial_scheduling(self.priority_list)

    def calculate_fitness_function(self, machine):
        """
        Function for calculating the fitness value of the current setting of y_jirm. After initializing the
        variables, equation 1, 5 and 6 are called to initially set the helper variables, which are then
        extended in the scheduling function in order to fulfill the prescribed constraints.

        Args:

        Returns:
            makespan (int): Makespan of the current setting of y_jirm
        """
        self.machine_time = np.zeros(
            (self.m, 1)
        )
        self.z_mu[machine:, :] = 0
        self.maintenance_duration = copy.deepcopy(self.initial_maintenance_duration)

        for m in range(machine, self.m):
            index_ji = []
            sort = []
            for j in range(self.j):
                index = np.nonzero(self.y_jirm[j][:, :, m])
                if len(index[0]) > 0:
                    index_ji.append([j, int(index[0]), int(index[1])])
                    sort.append(int(index[1]))
            index_ji = [y for x, y in sorted(zip(sort, index_ji))]

            for r in range(len(index_ji)):
                job = index_ji[r][0]
                operation = index_ji[r][1]
                self.time_calculation(m, job, operation)
                # check if maintenance has to be scheduled according to y_mwr
                for maintenance_w, t in enumerate(self.y_mwr[m]):
                    if t == r:
                        self.schedule_maintenance(m, operation, job, r, maintenance_w)

    def determine_termination_criterion(self):
        # TODO: Additional termination criterion, if no further convergence of the objective function
        self.iteration_number += 1
        if self.max_iter <= self.iteration_number - 1:
            return True
        return False

    def generate_new_solution(self):
        self._selecting_move_type()
        self._apply_move_type()

    def _selecting_move_type(self):
        if self.iteration_number % 1 == 0:
            self._move_maintenance_swap()
        elif self.iteration_number % 3 == 0:
            self._move_operation_insert_operation_on_one_machine()
        else:
            self._move_operation_position_swap_on_one_machine()

    def _apply_move_type(self):
        pass

    def _move_operation_insert_on_another_machine(self):
        """
        Increasing makespan is accepted in this move type
        """
        initial_y_jirm = copy.deepcopy(self.y_jirm)
        new_y = copy.deepcopy(self.y_jirm)
        selection = []
        makespan = []
        decision = []
        new_solution = None
        # Determine operations able to be processed on multiple machines
        for j in range(self.j):
            for i in range(self.i[j]):
                indices = np.nonzero(self.processing_time[j][i, :])
                if np.size(indices[0]) > 1:
                    selection.append([j, i, indices])
        # Randomly select i,j
        # TODO: Find a better way here
        try:
            choice = np.random.randint(0, len(selection))
        except:
            return None
        j = (selection[choice])[0]
        i = (selection[choice])[1]
        # Get current machine
        current_machine = int(np.where(self.y_jirm[j][i, :, :] == 1)[1])
        current_position = int(np.where(self.y_jirm[j][i, :, :] == 1)[0])
        # Randomly select new machine
        machines = ((selection[choice])[2])[0]
        while True:
            new_machine = machines[np.random.choice(len(machines))]
            if new_machine != current_machine:
                break
        # Try every position on new machine and select the one with smallest makespan
        r_max = self._determine_max_position(new_machine) + 1

        for r_new in range(
            r_max
        ):  # +1 because operation can also be set to the last position on new machine
            self.y_jirm = copy.deepcopy(initial_y_jirm)
            for r in range(self.r):
                for j in range(self.j):
                    if r == r_new:
                        new_y[j][:, r, new_machine] = self.y_jirm[j][
                            :, current_position, current_machine
                        ]
                    elif r > r_new:
                        new_y[j][:, r, new_machine] = self.y_jirm[j][
                            :, r - 1, new_machine
                        ]
                    else:
                        new_y[j][:, r, new_machine] = self.y_jirm[j][:, r, new_machine]

            for r in range(self.r):
                for j in range(self.j):
                    if r == self.r - 1:
                        new_y[j][:, r, current_machine] = 0
                    elif r >= current_position:
                        new_y[j][:, r, current_machine] = self.y_jirm[j][
                            :, r + 1, current_machine
                        ]
                    else:
                        new_y[j][:, r, current_machine] = self.y_jirm[j][
                            :, r, current_machine
                        ]
            # Generate new y_jirm
            self.y_jirm = copy.deepcopy(new_y)
            decision.append(self.calculate_fitness_function())
            # if self._check_feasibility() is not True:
            #    raise ValueError('Generated solution is not feasible!!!')
            makespan.append([decision[-1], new_y])
            # Some swaps are not feasible due to deadlock effects. Thus, invalid solutions occur
            print("Makespan of new machine assignment: {}".format(makespan[-1][0]))

            # if makespan smaller than previous best, update y
            if makespan[-1][0] <= self.make_span[-1]:
                self.make_span.append(makespan[-1][0])
                self.optimized_solution = copy.deepcopy(new_y)
                new_solution = copy.deepcopy(new_y)
                minimum = int(np.argmin(decision))
                self.y_jirm = copy.deepcopy((makespan[minimum])[1])
        # Calculate new makespan
        if new_solution is not None:
            self.y_jirm = copy.deepcopy(new_solution)
        else:
            self.y_jirm = copy.deepcopy(initial_y_jirm)

    def _move_operation_position_swap_on_one_machine(self):
        """
        Perform a swapping move type on a randomly selected machine. On this machine, one operation i.e. one position
        is selected randomly and it will change position with all other operations on this machine ona after the other.
        This generates the neighborhood set. If a new neighbor is better than the currently best, the decision variable
        will be updated.
        """
        new_y = None
        initial_y_jirm = copy.deepcopy(self.y_jirm)
        # Change y_jirm and calculate all other helper variables again and determine makespan
        # select the machine to perform this operation
        max_pos = 0
        m = np.random.randint(0, self.m)
        max_pos = self._determine_max_position(m)
        if max_pos <= 1:
            return None
        r_swap = np.random.randint(0, max_pos)
        # swap this position with all others, this gives the neighborhood
        for r in range(max_pos):
            if r != r_swap:  # TODO: Additionally, check if move is in tabu list
                # reset self.y_jirm
                self.y_jirm = copy.deepcopy(initial_y_jirm)
                for j in range(self.j):
                    # swap self.y_jirm[:,:,r,m] with self.y_jirm[:,:,r_swap,m]
                    swap = np.copy(self.y_jirm[j][:, r_swap, m])
                    self.y_jirm[j][:, r_swap, m] = self.y_jirm[j][:, r, m]
                    self.y_jirm[j][:, r, m] = swap

                makespan = self.calculate_fitness_function()
                # Some swaps are not feasible due to deadlock effects. Thus, invalid solutions occur
                print("Makespan of new neighbor: {}".format(makespan))

                # if makespan smaller than previous best, update y
                if makespan <= self.make_span[-1]:
                    self.make_span.append(makespan)
                    new_y = copy.deepcopy(self.y_jirm)
                    self.optimized_solution = copy.deepcopy(new_y)
                    # self.plot_gantt_chart(
                    #    group=True,
                    #    title="Swap {} and {} on machine {}, Feasibility: {}".format(r_swap, r, m, self.feasibility),
                    # )
        # Set y_jirm to the generated optimal solution if available
        if new_y is not None:
            self.y_jirm = copy.deepcopy(new_y)
        else:
            self.y_jirm = copy.deepcopy(initial_y_jirm)
        # TODO: Selection of the best solution, y_jirm = best solution
        # TODO: Add move type to the tabu list

    def _move_operation_insert_operation_on_one_machine(self):
        new_solution, new_y = None, None
        initial_y_jirm = copy.deepcopy(self.y_jirm)
        # Change y_jirm and calculate all other helper variables again and determine makespan
        # select the machine to perform this operation
        m = np.random.randint(0, self.m)
        max_pos = self._determine_max_position(m)
        if max_pos <= 1:
            return None
        r_insert = np.random.randint(0, max_pos)
        # swap this position with all others, this gives the neighborhood
        for r in range(max_pos):
            if r != r_insert:  # TODO: Additionally, check if move is in tabu list
                # reset self.y_jirm
                self.y_jirm = copy.deepcopy(initial_y_jirm)
                new_y = copy.deepcopy(initial_y_jirm)
                for j in range(self.j):
                    for rr in range(self.r):
                        if rr == r:
                            new_y[j][:, rr, m] = self.y_jirm[j][:, r_insert, m]
                        elif r_insert <= rr < r:  # Movement to right hand side
                            new_y[j][:, rr, m] = self.y_jirm[j][:, rr + 1, m]
                        elif r < rr <= r_insert:  # Movement to left hand side
                            new_y[j][:, rr, m] = self.y_jirm[j][:, rr - 1, m]
                        else:
                            new_y[j][:, rr, m] = self.y_jirm[j][:, rr, m]
                if new_y is None:
                    return None
                else:
                    self.y_jirm = copy.deepcopy(new_y)

                makespan = self.calculate_fitness_function()
                # Some inserts are not feasible due to deadlock effects. Thus, invalid solutions occur
                print("Makespan of new neighbor: {}".format(makespan))

                # if makespan smaller than previous best, update y
                if makespan <= self.make_span[-1]:
                    self.make_span.append(makespan)
                    new_solution = copy.deepcopy(self.y_jirm)
                    self.optimized_solution = copy.deepcopy(new_solution)
                    # self.plot_gantt_chart(
                    #    group=True,
                    #    title="Insert {} to position {} on machine {}, Feasibility: {}".format(r_insert, r, m, self.feasibility),
                    # )
        if new_solution is not None:
            self.y_jirm = copy.deepcopy(new_solution)
        else:
            self.y_jirm = copy.deepcopy(initial_y_jirm)
        # TODO: Selection of the best solution, y_jirm = best solution
        # TODO: Add move type to the tabu list

    def _move_maintenance_swap(self):
        new_solution, new_y_mwr = None, None
        initial_y_mwr = copy.deepcopy(self.y_mwr)
        # select a feasible machine
        machine_select = []
        for m in range(len(self.y_mwr)):
            if self.y_mwr[m]:
                machine_select.append(m)
        machine = np.random.choice(machine_select)
        max_pos = self._determine_max_position(machine) - 2
        if max_pos <= 1:
            return None
        maintenance_w = np.random.randint(0, len(self.y_mwr[machine]))
        current_position = self.y_mwr[machine][maintenance_w]

        for r in range(max_pos):
            if r != current_position:
                self.y_mwr[machine][maintenance_w] = r

                self.calculate_fitness_function(machine)
                makespan = []
                for j in range(self.j):
                    makespan.append(np.max(self.completion_time_ij[j]))
                self.makespan = max(makespan)
                print('Makespan: {}'.format(self.makespan))

                if self.makespan < self.make_span[-1]:
                    self.make_span.append(self.makespan)
                    self.optimized_solution = copy.deepcopy([self.y_jirm, self.y_mwr])
                    new_y_mwr = copy.deepcopy(self.optimized_solution[1])

                self.y_mwr = copy.deepcopy(initial_y_mwr)
        if new_y_mwr is not None:
            self.y_mwr = copy.deepcopy(new_y_mwr)

    def _move_maintenance_ii(self):
        pass

    def _move_buffer_i(self):
        pass

    def _move_buffer_ii(self):
        pass

    def _determine_max_position(self, machine):
        max_pos = []
        for j in range(self.j):
            index = np.where(self.y_jirm[j][:, :, machine] == 1)[1]
            if index.size != 0:
                max_pos.append(int(index) + 1)
        if len(max_pos) == 0:
            max_pos = 1
        else:
            max_pos = np.max(max_pos)
        return max_pos

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

        self.j = np.shape(self.processing_time)[0]
        self.m = np.shape(self.processing_time)[2]
        self.r = (
            int(
                np.max(
                    np.sum(
                        np.sum(np.array(self.processing_time, dtype=bool), axis=0),
                        axis=0,
                    )
                )
            )
            * 3
        )

        self.i = []
        new_processing_time = []
        for j in range(self.j):
            processing_time = self.processing_time[j, :, :]
            self.i.append(
                np.shape(processing_time[~np.all(processing_time == 0, axis=1)])[0]
            )
            new_processing_time.append(
                processing_time[~np.all(processing_time == 0, axis=1)]
            )
        self.processing_time = copy.deepcopy(new_processing_time)
        # self.shape = [self.j, self.i, self.r, self.m]
        self.colors = self._generate_colors()

        if self.maintenance_duration is not None:
            self.z_mu = np.zeros((self.m, 9000))
        return initial_length

    def initialize_times(self):
        self.starting_time_rm = np.zeros((self.r, self.m))
        self.starting_time_rm[0, :] = 0
        self.completion_time_rm = np.zeros((self.r, self.m))
        self.starting_time_ij = []  # np.zeros((self.i, self.j))
        self.completion_time_ij = []  # np.zeros((self.i, self.j))
        # self.starting_time_mw = copy.deepcopy(self.maintenance_duration)
        # self.completion_time_mw = copy.deepcopy(self.maintenance_duration)
        self.starting_time_mw = [[] for _ in np.arange(len(self.maintenance_duration))]
        self.completion_time_mw = [
            [] for _ in np.arange(len(self.maintenance_duration))
        ]
        self.y_mwr = [[] for _ in np.arange(len(self.maintenance_duration))]

        for j in range(self.j):
            self.starting_time_ij.append(np.zeros((self.i[j])))
            self.completion_time_ij.append(np.zeros((self.i[j])))

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
        self.initialize_times()

        p_ji = []
        for j in range(self.j):
            p_ji.append(np.zeros((self.i[j])))

        weights = []  # initialize weights list
        # loop over all indices of processing time
        for j in range(self.j):
            for i in range(self.i[j]):
                for m in range(self.m):
                    weights.append(self.processing_time[j][i, m])
                weights = weights / np.sum(
                    weights
                )  # normalize weights, that sum(weights) = 1
                indices = np.arange(self.m)  # machine indices from 0 to self.m

                machine_index = np.random.choice(
                    indices, p=weights
                )  # randomly select a machine, w.r.t. weights
                p_ji[j][i] = self.processing_time[j][
                    i, machine_index
                ]  # store processing time in reduced array
                weights = []  # reset weights
        #  --------------------------------------------------------------------
        total_time = []
        for j in range(self.j):
            total_time.append(-np.sum(p_ji[j]))

        # total_time = np.sum(
        #    p_ji, axis=0
        # )  # calculate total completion time for each job (sum over i)
        job_index = np.arange(self.j)  # job indices from 0 to self.j
        sort = np.argsort(
            total_time
        )  # get indices of sorted processing time (no explicit tiebreaker considered)
        priority_list = np.column_stack(
            (job_index[sort], np.array(total_time)[sort])
        )  # generate priority list

        print("Average lower bound for makespan: {}".format(priority_list[-1][1]))

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
        self.y_jirm = []  # initialize decision variable
        self.machine_time = np.zeros(
            (self.m, 1)
        )  # gets updated with scheduled operation and maintenance
        for j in range(self.j):
            self.y_jirm.append(np.zeros((self.i[j], self.r, self.m)))

        self.machine_position = np.array(
            [0] * self.m
        )  # list to store used machine positions
        self.schedule_operation(
            int(priority_list[0, 0])
        )  # schedule first job on all machines

        for job_index in priority_list[1:, 0]:  # loop over priority list
            job_index = int(job_index)
            self.schedule_operation(job_index)

        makespan = []
        for j in range(self.j):
            makespan.append(np.max(self.completion_time_ij[j]))
        self.makespan = max(makespan)

    def schedule_operation(self, job_index):
        for i in range(self.i[job_index]):  # loop over operations
            machine_index = []  # reset machine index
            for m in range(self.m):  # loop over machine
                if self.processing_time[job_index][i, m] > 0:
                    machine_index.append(m)
            if not machine_index:
                machine_choice = 0
                warnings.warn("Operation has no machine assigned to it!")
            else:
                machine_choice = machine_index[
                    np.argmin(
                        self.machine_position[machine_index]
                    )  # get the minimum of machine position and choose this machine
                ]
            self.time_calculation(machine_choice, job_index, i)
            self.schedule_maintenance(machine_choice, i, job_index, self.machine_position[machine_choice])
            self.y_jirm[job_index][
                i, self.machine_position[machine_choice], machine_choice
            ] = 1
            self.machine_position[machine_choice] += 1

    def time_calculation(self, machine, job, operation):
        self.starting_time_ij[job][operation] = copy.deepcopy(
            self.machine_time[machine]
        )
        self.completion_time_ij[job][operation] = copy.deepcopy(
            self.machine_time[machine] + self.processing_time[job][operation, machine]
        )
        if operation > 0:
            diff = (
                self.completion_time_ij[job][operation - 1]
                - self.starting_time_ij[job][operation]
            )
            if diff > 0:
                self.starting_time_ij[job][operation] = copy.deepcopy(self.starting_time_ij[job][operation] + diff)
                self.completion_time_ij[job][operation] = copy.deepcopy(self.completion_time_ij[job][operation] + diff)

        self.machine_time[machine] = copy.deepcopy(
            self.completion_time_ij[job][operation]
        )

    def schedule_maintenance(self, machine, operation, job, position, position_w=None):
        # check if maintenance is necessary for this machine
        if self.maintenance_duration[machine] is not None:
            for maintenance_w in range(len(self.maintenance_duration[machine][:])):
                if position_w is None or maintenance_w == position_w:
                    if self.maintenance_duration[machine][maintenance_w] != 0:
                        if position_w is None:
                            self.y_mwr[machine].append(position)

                        start = copy.deepcopy(self.machine_time[machine][0])
                        period = 10
                        periods = 0
                        max_simultaneous = 3
                        required_periods = self.maintenance_duration[machine][maintenance_w]/period
                        earliest_start = start
                        sum_z_mu = np.sum(self.z_mu, axis=0)
                        for p in range(int(start/period), 9000):
                            # count number of maintenance in this period p:
                            simultaneous_maintenance = sum_z_mu[p]  # in period p
                            if simultaneous_maintenance >= max_simultaneous:
                                earliest_start = (p + 1) * period
                                periods = 0
                            else:
                                periods += 1
                            if (
                                periods >= required_periods
                            ):
                                break
                        if position_w is None:
                            self.starting_time_mw[machine].append(earliest_start)
                            self.completion_time_mw[machine].append(
                                earliest_start + self.maintenance_duration[machine][maintenance_w]
                            )
                        else:
                            self.starting_time_mw[machine][maintenance_w] = copy.deepcopy(earliest_start)
                            self.completion_time_mw[machine][maintenance_w] = copy.deepcopy(
                                earliest_start + self.maintenance_duration[machine][maintenance_w]
                            )
                        machine_time = copy.deepcopy(self.maintenance_duration[machine][maintenance_w])
                        self.machine_time[machine] = earliest_start + machine_time
                        self.maintenance_duration[machine][maintenance_w] = 0
                        # generate z_mu:
                        start_index = int(self.starting_time_mw[machine][maintenance_w] / period)
                        end_index = int(self.completion_time_mw[machine][maintenance_w] / period)
                        self.z_mu[machine, start_index:end_index] += 1
                        if any(self.z_mu[machine, :] > 1):
                            raise ValueError('Maintenance overlap!!! Abort...')
                        break
        else:
            self.machine_time[machine] = copy.deepcopy(
                self.completion_time_ij[job][operation]
            )

    def plot_gantt_chart(self, group=False, title="Gantt_Chart"):
        """
        Plot Gantt Chart depending on the starting and completion times of all operations of each job

        Args:
            self.starting_time_ij (array):
            self.completion_time_ij (array):
            group (bool): False if machines in gantt chart are not supposed to group up, True if they are supposed
            to be grouped.
            title (str): Title of plot

        Returns:

        """

        import plotly.figure_factory as ff
        import plotly.io as pio

        pio.renderers.default = "browser"
        df = []

        for j in range(self.j):
            for m in range(self.m):
                for i in range(self.i[j]):
                    if any(np.array(self.y_jirm[j][i, :, m], dtype=bool)) is True:
                        entry = dict(
                            Task=str(m),
                            Start=str(float(self.starting_time_ij[j][i])),
                            Finish=str(float(self.completion_time_ij[j][i])),
                            Resource="job " + str(j),
                        )
                        df.append(entry)

        for m in range(self.m):
            if self.starting_time_mw[m] is not []:
                for w in range(len(self.starting_time_mw[m])):
                    entry = dict(
                        Task=str(m),
                        Start=str(float(self.starting_time_mw[m][w])),
                        Finish=str(float(self.completion_time_mw[m][w])),
                        Resource="Maintenance",
                    )
                    df.append(entry)

        fig = ff.create_gantt(
            df,
            colors=self.colors,
            index_col="Resource",
            show_colorbar=True,
            group_tasks=group,
            title=title,
        )
        fig.update_traces(
            mode="lines", line_color="black", selector=dict(fill="toself")
        )

        fig.show()

    def _generate_colors(self):
        colors = []
        for j in range(self.j):
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            colors.append("#%02X%02X%02X" % (r, g, b))

        if self.maintenance_duration is not None:
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            colors.append("#%02X%02X%02X" % (r, g, b))
        return colors


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

    # processing_time_path = "/Users/q517174/PycharmProjects/Optiflex/processing_time.csv"
    # processing_time_input = extract_csv(processing_time_path, 3)

    processing_time_path = "parameter/Takzeit_overview.xlsx"
    variants_of_interest = ["B37 D", "B37 C15 TUE1", "B48 B20 TUE1", "B38 A15 TUE1"]
    amount_of_variants = [10, 10, 10, 10]
    processing_time_input, maintenance = extract_parameter(
        processing_time_path, variants_of_interest, amount_of_variants, maintenance=True
    )

    # After read in everything, the object can be created and the main_run can start
    job_shop_object = JobShop(
        processing_time=processing_time_input,
        maintenance_duration=maintenance,
        max_iter=100,
    )
    job_shop_object.main_run()
