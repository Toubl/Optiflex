import pickle
import numpy as np
import pandas as pd
from extract_csv import extract_parameter
from fitness_function import FitnessFunction
import matplotlib.pyplot as plt
from operator import truediv

"""Script for output data evaluation and visualization"""

pickle_folder = 'output_pickle/'
study_name = 'study24dec'
pickle_file = pickle_folder + study_name + '.pickle'
amount_of_variants = [10, 100, 60, 30]
maintenance_time = 720
processing_time_path = "parameter/Takzeit_overview.xlsx"
variants_of_interest = ["B37 D", "B37 C15 TUE1", "B48 B20 TUE1", "B38 A15 TUE1"]
processing_time_input, maintenance = extract_parameter(
    processing_time_path, variants_of_interest, amount_of_variants, maintenance=True
)

# read in pickle file
pickle_data = []
with (open(pickle_file, 'rb')) as study:
    while True:
        try:
            pickle_data.append(pickle.load(study))
        except EOFError:
            break

input_data = pickle_data[0]['Optimum']
del pickle_data

fitness_function_object = FitnessFunction(
    processing_time=processing_time_input,
    maintenance_duration=maintenance,
    max_simultaneous=6,
)

y_jirm = input_data[0]
y_mwr = input_data[1]

fitness_function_object.calculate(plot=False, y_jirm=y_jirm, y_mwr=y_mwr)

amount_m = fitness_function_object.m
starting_time_ij = fitness_function_object.starting_time_ij
completion_time_ij = fitness_function_object.completion_time_ij
starting_time_mw = fitness_function_object.starting_time_mw
completion_time_mw = fitness_function_object.completion_time_mw
i_j = fitness_function_object.i
z_mu = fitness_function_object.z_mu
makespan = fitness_function_object.makespan


def calculate_starving(amount_m, starting_time_ij, completion_time_ij, starting_time_mw, completion_time_mw, y_jirm,
                       y_mwr, maintenance_time, i_j, study_name, z_mu, makespan):
    afo_list = [
        "AFO005",
        "AFO010",
        "AFO020",
        "AFO040",
        "AFO050",
        "AFO060",
        "AFO070",
        "AFO080",
        "AFO090",
        "AFO100",
        "AFO110",
        "AFO130",
        "AFO140",
        "AFO150",
        "AFO155",
        "AFO160",
        "AFO170",
        "AFO190",
    ]
    starving = []
    starving_rel = []
    starving_maint = []
    number_of_jobs = []
    mach = []
    mean_time = []
    processing_time = []
    makespan_machine = []
    for machine in range(amount_m):
        index_ji = []
        sort = []
        for j in range(200):
            index = np.nonzero(y_jirm[j][:, :, machine])
            if len(index[0]) > 0:
                index_ji.append([j, int(index[0]), int(index[1])])
                sort.append(int(index[1]))
        index_ji = [y for x, y in sorted(zip(sort, index_ji))]

        if len(index_ji) > 0:
            number_of_jobs.append(len(index_ji))

            end = completion_time_ij[index_ji[-1][0]][index_ji[-1][1]]
            start = starting_time_ij[index_ji[0][0]][index_ji[0][1]]

            processtime = 0
            for n, val in enumerate(index_ji):
                j = val[0]
                i = val[1]
                processtime += (completion_time_ij[j][i] - starting_time_ij[j][i])

            processing_time.append(processtime)
            starving.append(end - start - processtime)  # needed
            makespan_machine.append(end - start)  # needed
            starving_rel.append((end - start - processtime) / (end - start))
            starving_maint.append((end - start - processtime - maintenance_time) / (end - start))
            mach.append(machine)
            mean_time.append(processtime / len(index_ji))
        else:
            starving.append(None)
            makespan_machine.append(None)

    """
    Precedence machine with longest waiting time within one job
    """
    waiting_on_machine = [[] for x in range(amount_m)]
    for j in range(200):
        for i in range(i_j[j] - 1):
            waiting_time = starting_time_ij[j][i + 1] - completion_time_ij[j][i]
            position = np.nonzero(y_jirm[j][i, :, :])
            machine = position[1][0]
            waiting_on_machine[machine].append(waiting_time)

    waiting_average = []
    for m in range(amount_m):
        jobs = 0
        count = 0
        if len(waiting_on_machine[m]) > 0:
            for i in range(len(waiting_on_machine[m])):
                count += waiting_on_machine[m][i]
                jobs += 1
            waiting_average.append(count/jobs)
        else:
            waiting_average.append(None)


    var = [] * amount_m
    mu = [] * amount_m
    for m in range(amount_m):
        variation = 0
        if len(waiting_on_machine[m]) > 0:
            mu.append(sum(waiting_on_machine[m]) / len(waiting_on_machine[m]))

            for entries in range(len(waiting_on_machine[m])):
                variation += (waiting_on_machine[m][entries] - mu[m]) ** 2
            var.append(variation / len(waiting_on_machine[m]))
        else:
            mu.append(0)
            var.append(None)

    vark = [] * amount_m
    for m in range(amount_m):
        if mu[m] != 0:
            vark.append(np.sqrt(var[m]) / mu[m])
        else:
            vark.append(None)

    output = pd.DataFrame(
        {'Waiting time': waiting_average,
         'Variationkoeff': vark,
         'Starving time': starving,
         'Makespan on machine': makespan_machine
         })

    # save as excel
    filename = 'figures/' + study_name + '_evaluation_talbe.xlsx'
    output.to_excel(filename, index=True, header=True)


calculate_starving(amount_m=amount_m, starting_time_ij=starting_time_ij,
                   completion_time_ij=completion_time_ij,
                   starting_time_mw=starting_time_mw, completion_time_mw=completion_time_mw,
                   y_jirm=y_jirm, y_mwr=y_mwr, maintenance_time=maintenance_time, i_j=i_j, study_name=study_name,
                   z_mu=z_mu, makespan=makespan)
