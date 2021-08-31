import pickle
import numpy as np
import pandas as pd
import copy
import warnings


class FitnessFunction:

    def __init__(
            self,
            maintenance_max=None,
            processing_time=None,
            maintenance_duration=None,
            n_m=None,
            buffer_capacity=None,
            max_iter=None,
            max_simultaneous=None,
            convergence_limit=50,
            insert_another=2,
            maintenance_swap=4,
            insert_one=2,
            operation_swap=2,
            period=10,
    ):
        self.maintenance_max = maintenance_max
        self.init_processing_time = processing_time
        self.maintenance_duration = maintenance_duration
        self.n_m = n_m
        self.buffer_capacity = buffer_capacity
        self.max_iter = max_iter
        self.Q = max_simultaneous
        self.convergence_limit = convergence_limit
        self.insert_another = insert_another
        self.insert_one = insert_one
        self.maintenance_swap = maintenance_swap
        self.operation_swap = operation_swap
        self.period = period

        self.processing_time = None
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
        self.y_mwr = None
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

        self.initial_maintenance_duration = copy.deepcopy(maintenance_duration)
        self.multiple_machines = None
        self.move_type_selection = {
            "maintenance_swap": 0,
            "insert_one": 0,
            "insert_another": 0,
            "operation_swap": 0,
        }
        self.improvement = None
        self.final_best = None
        self.output_dict = {
            "Makespan": [],
            "Movetype": [],
            "Iteration": [],
            "Optimum": [],
            "Initial": [],
        }

        self.feasibility = None
        self.start_time = None

    def calculate(self, y_jirm, y_mwr, plot=False):
        self._estimating_initial_domains()
        self.initialize_times()

        self.y_jirm = copy.deepcopy(y_jirm)
        self.y_mwr = copy.deepcopy(y_mwr)
        self.calculate_fitness_function()
        self._calculate_makespan()

        if plot is True:
            self.colors = self._generate_colors()
            self.plot_gantt_chart(group=True, title='Gantt Chart')

    def calculate_fitness_function(self, machine=0):
        """
        Function for calculating the fitness value of the current setting of y_jirm. After initializing the
        variables, equation 1, 5 and 6 are called to initially set the helper variables, which are then
        extended in the scheduling function in order to fulfill the prescribed constraints.

        Args:

        Returns:
            makespan (int): Makespan of the current setting of y_jirm
        """
        self.machine_time = np.zeros((self.m, 1))
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

        # initial_length = np.sum(self.processing_time)

        self.j = np.shape(self.init_processing_time)[0]
        self.m = np.shape(self.init_processing_time)[2]
        self.r = (
                int(
                    np.max(
                        np.sum(
                            np.sum(np.array(self.init_processing_time, dtype=bool), axis=0),
                            axis=0,
                        )
                    )
                )
                * 3
        )

        self.i = []
        new_processing_time = []
        for j in range(self.j):
            processing_time = self.init_processing_time[j, :, :]
            self.i.append(
                np.shape(processing_time[~np.all(processing_time == 0, axis=1)])[0]
            )
            new_processing_time.append(
                processing_time[~np.all(processing_time == 0, axis=1)]
            )
        self.processing_time = copy.deepcopy(new_processing_time)
        # self.shape = [self.j, self.i, self.r, self.m]

        if self.maintenance_duration is not None:
            self.z_mu = np.zeros((self.m, 9000))

    def initialize_times(self):
        self.starting_time_rm = np.zeros((self.r, self.m))
        self.starting_time_rm[0, :] = 0
        self.completion_time_rm = np.zeros((self.r, self.m))
        self.starting_time_ij = []  # np.zeros((self.i, self.j))
        self.completion_time_ij = []  # np.zeros((self.i, self.j))
        # self.starting_time_mw = copy.deepcopy(self.maintenance_duration)
        # self.completion_time_mw = copy.deepcopy(self.maintenance_duration)
        self.starting_time_mw = copy.deepcopy(
            self.maintenance_duration
        )  # [[] for _ in np.arange(len(self.maintenance_duration))]
        self.completion_time_mw = copy.deepcopy(self.maintenance_duration)  # [
        #    [] for _ in np.arange(len(self.maintenance_duration))
        # ]
        self.y_mwr = copy.deepcopy(
            self.maintenance_duration
        )  # [[] for _ in np.arange(len(self.maintenance_duration))]

        for j in range(self.j):
            self.starting_time_ij.append(np.zeros((self.i[j])))
            self.completion_time_ij.append(np.zeros((self.i[j])))

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
                self.starting_time_ij[job][operation] = copy.deepcopy(
                    self.starting_time_ij[job][operation] + diff
                )
                self.completion_time_ij[job][operation] = copy.deepcopy(
                    self.completion_time_ij[job][operation] + diff
                )

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
                        periods = 0
                        required_periods = (
                                self.maintenance_duration[machine][maintenance_w]
                                / self.period
                        )
                        earliest_start = start
                        sum_z_mu = np.sum(self.z_mu, axis=0)
                        for p in range(int(start / self.period), 9000):
                            # count number of maintenance in this period p:
                            simultaneous_maintenance = sum_z_mu[p]  # in period p
                            if simultaneous_maintenance >= self.Q:
                                earliest_start = (p + 1) * self.period
                                periods = 0
                            else:
                                periods += 1
                            if periods >= required_periods:
                                break
                        if position_w is None:
                            self.starting_time_mw[machine].append(earliest_start)
                            self.completion_time_mw[machine].append(
                                earliest_start
                                + self.maintenance_duration[machine][maintenance_w]
                            )
                        else:
                            self.starting_time_mw[machine][
                                maintenance_w
                            ] = copy.deepcopy(earliest_start)
                            self.completion_time_mw[machine][
                                maintenance_w
                            ] = copy.deepcopy(
                                earliest_start
                                + self.maintenance_duration[machine][maintenance_w]
                            )
                        machine_time = copy.deepcopy(
                            self.maintenance_duration[machine][maintenance_w]
                        )
                        self.machine_time[machine] = earliest_start + machine_time
                        self.maintenance_duration[machine][maintenance_w] = 0
                        # generate z_mu:
                        start_index = int(
                            self.starting_time_mw[machine][maintenance_w] / self.period
                        )
                        end_index = int(
                            self.completion_time_mw[machine][maintenance_w]
                            / self.period
                        )
                        self.z_mu[machine, start_index:end_index] += 1
                        if any(self.z_mu[machine, :] > 1):
                            raise ValueError("Maintenance overlap!!! Abort...")
                        break
        else:
            self.machine_time[machine] = copy.deepcopy(
                self.completion_time_ij[job][operation]
            )

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
        job_index = np.arange(self.j)  # job indices from 0 to self.j
        sort = np.argsort(
            total_time
        )  # get indices of sorted processing time (no explicit tiebreaker considered)
        priority_list = np.column_stack(
            (job_index[sort], np.array(total_time)[sort])
        )  # generate priority list

        # print("Average lower bound for makespan: {}".format(-priority_list[-1][1]))
        return priority_list

    def _random_maintenance_for_initialization_scaled(self):
        """
        Can be used for diversification if the optimization seems to be trapped in a local minimum.
        """
        self.maintenance_duration = copy.deepcopy(self.initial_maintenance_duration)
        for machine in range(self.m):
            max_position = self._determine_max_position(machine)
            # check if maintenance is on this machine
            if self.maintenance_duration[machine] is not None:
                for maintenance_w in range(len(self.maintenance_duration[machine][:])):
                    if max_position <= 1:
                        position = 0
                    else:
                        position = np.random.randint(0, max_position - 1)
                    a = np.arange(int((max_position - 2) * (machine / self.m)) + 1)
                    self.y_mwr[machine][maintenance_w] = np.random.choice(a)

    def _random_maintenance_for_initialization_unscaled(self):
        """
        Can be used for diversification if the optimization seems to be trapped in a local minimum.
        """
        self.maintenance_duration = copy.deepcopy(self.initial_maintenance_duration)
        for machine in range(self.m):
            max_position = self._determine_max_position(machine)
            # check if maintenance is on this machine
            if self.maintenance_duration[machine] is not None:
                for maintenance_w in range(len(self.maintenance_duration[machine][:])):
                    if max_position <= 1:
                        position = 0
                    else:
                        position = np.random.randint(0, max_position - 1)
                    self.y_mwr[machine][maintenance_w] = position

    def _initial_maintenance(self):
        """
        Can be used for diversification if the optimization seems to be trapped in a local minimum.
        """
        self.maintenance_duration = copy.deepcopy(self.initial_maintenance_duration)
        for machine in range(self.m):
            # max_position = self._determine_max_position(machine)
            # check if maintenance is on this machine
            if self.maintenance_duration[machine] is not None:
                for maintenance_w in range(len(self.maintenance_duration[machine][:])):
                    self.y_mwr[machine][maintenance_w] = 0

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
        self._initial_maintenance()
        self.calculate_fitness_function(0)

    def _calculate_makespan(self):
        makespan = []
        for j in range(self.j):
            makespan.append(np.max(self.completion_time_ij[j]))
        self.makespan = max(makespan)

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

        for m in range(self.m):
            if self.starting_time_mw[m] is not None:
                for w in range(len(self.starting_time_mw[m])):
                    entry = dict(
                        Task=str(m),
                        Start=str(float(self.starting_time_mw[m][w])),
                        Finish=str(float(self.completion_time_mw[m][w])),
                        Resource="Maintenance",
                    )
                    df.append(entry)

            for j in range(self.j):
                for i in range(self.i[j]):
                    if any(np.array(self.y_jirm[j][i, :, m], dtype=bool)) is True:
                        entry = dict(
                            Task=str(m),
                            Start=str(float(self.starting_time_ij[j][i])),
                            Finish=str(float(self.completion_time_ij[j][i])),
                            Resource="job " + str(j),
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
