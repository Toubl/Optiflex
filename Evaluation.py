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

fitness_function_object.calculate(plot=True, y_jirm=y_jirm, y_mwr=y_mwr)

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
            starving.append(end - start - processtime)
            starving_rel.append((end - start - processtime) / (end - start))
            starving_maint.append((end - start - processtime - maintenance_time) / (end - start))
            mach.append(machine)
            mean_time.append(processtime / len(index_ji))

    """
    Number of jobs assigned to each machine in an AFO
    """
    fig, axs = plt.subplots(3, 6, figsize=(20, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.1, wspace=.1)
    axs = axs.ravel()

    afo_number_of_jobs = [[] for i in range(18)]
    machine_names = [[] for i in range(18)]
    for i in range(len(number_of_jobs)):
        afo = int(mach[i] / 7)
        afo_number_of_jobs[afo].append(number_of_jobs[i])
        machine_names[afo].append('M' + str(mach[i] % 7 + 1))
    for i in range(18):
        y = np.array(afo_number_of_jobs[i])
        axs[i].pie(y, labels=machine_names[i])
        axs[i].set_title(afo_list[i])
    fig.suptitle('Number of jobs assigned to each machine in an AFO, varies with product mix', fontsize=30)
    plt.savefig('figures/' + study_name + '_number_of_jobs.pdf')

    """
    Mean processing time over all machines in an AFO
    """
    fig, axs = plt.subplots(3, 6, figsize=(20, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.1, wspace=.1)
    axs = axs.ravel()

    afo_processing_time = [[] for i in range(18)]
    for i in range(len(processing_time)):
        afo = int(mach[i] / 7)
        afo_processing_time[afo].append(processing_time[i] / number_of_jobs[i])
    for i in range(18):
        y = np.array(afo_processing_time[i])
        total = sum(y)
        axs[i].pie(y, labels=machine_names[i], autopct=lambda p: '{:.0f}'.format(p * total / 100))
        axs[i].set_title(afo_list[i])
    fig.suptitle('Mean processing time over all machines in an AFO', fontsize=30)
    plt.savefig('figures/' + study_name + '_mean_processing_time.pdf')

    """
    Starving time for all machines in an AFO
    """
    fig, axs = plt.subplots(3, 6, figsize=(20, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.1, wspace=.1)
    axs = axs.ravel()

    afo_starving_time = [[] for i in range(18)]
    for i in range(len(processing_time)):
        afo = int(mach[i] / 7)
        afo_starving_time[afo].append(starving[i])
    for i in range(18):
        y = np.array(afo_starving_time[i])
        total = sum(y)
        axs[i].pie(y, labels=machine_names[i], autopct=lambda p: '{:.0f}'.format(p * total / 100))
        axs[i].set_title(afo_list[i])
    fig.suptitle('Starving time for all machines in an AFO', fontsize=30)
    plt.savefig('figures/' + study_name + '_starving_time.pdf')

    """
    Precedence machine with longest waiting time within one job
    """
    plt.figure(6)
    waiting_time = 0
    machine = None
    machines = []
    for j in range(200):
        for i in range(5, i_j[j] - 1):
            if starting_time_ij[j][i + 1] - completion_time_ij[j][i] > waiting_time:
                waiting_time = starting_time_ij[j][i + 1] - completion_time_ij[j][i]
                position = np.nonzero(y_jirm[j][i, :, :])
                machine = position[1][0]
        machine = int(machine / 7)
        machines.append(machine)
        waiting_time = 0

    bottleneck = [0] * 18
    for i, val in enumerate(machines):
        bottleneck[val] += 1

    plt.bar(afo_list, bottleneck)
    plt.ylabel('Amount of Jobs')
    plt.title('AFO with largest waiting time')
    plt.xticks(rotation=90)
    plt.savefig('figures/' + study_name + '_bottleneck.pdf')



    """
    Number of simultaneous maintenance
    """
    plt.figure(7)
    simultan = np.sum(z_mu, axis=0)
    simultan = simultan[:int(makespan / 10)]
    u = np.arange(int(makespan/10))
    plt.plot(u, simultan)
    plt.ylabel('Maintenance activities')
    plt.title('Number of simultaneous maintenance activities')
    plt.xlabel('Periods')
    plt.savefig('figures/' + study_name + '_maintenance_activities.pdf')

    """
    # Plotting part
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].bar(mach, number_of_jobs)
    axs[0, 0].set_title('Number of Jobs')
    axs[0, 0].set(xlabel='Machines', ylabel='Amount')

    axs[0, 1].bar(mach, starving)  # , 'tab:blue')
    ax_new = axs[0, 1].twinx()
    ax_new.plot(mach, mean_time, 'tab:orange')
    axs[0, 1].set_title('Starving Time (abs)')
    axs[0, 0].set(xlabel='Machines', ylabel='Time (Periods)')

    axs[1, 0].bar(mach, starving_rel)  # , 'tab:green')
    axs[1, 0].set_title('Starving Time (rel)')
    axs[0, 0].set(xlabel='Machines', ylabel='Percentage')

    axs[1, 1].bar(mach, starving_maint)  # , 'tab:red')
    axs[1, 1].set_title('Starving Time (ral, maintenance)')
    axs[0, 0].set(xlabel='Machines', ylabel='Percentage')

    fig.tight_layout()
    plt.show()

    # Contract machines to afo
    m_afo = [0] * 18
    t_afo = [0] * 18
    starving_afo = [0] * 18
    jobs_afo = [0] * 18
    for i in range(len(processing_time)):
        afo = int(mach[i] / 7)
        m_afo[afo] += 1
        t_afo[afo] += processing_time[i] / max(processing_time)
        starving_afo[afo] += starving_rel[i]
        jobs_afo[afo] += number_of_jobs[i]
    heat = map(truediv, t_afo, m_afo)
    starv_afo = list(map(truediv, starving_afo, m_afo))
    jobs_afo = list(map(truediv, jobs_afo, m_afo))
    jobs_afo = [x / max(jobs_afo) for x in jobs_afo]


    m_afo = [x / 7 for x in m_afo]
    df = pd.DataFrame({'Machines': m_afo, 'Processing Time': heat, 'Starving Time': starv_afo, 'Jobs': jobs_afo},
                      index=afo_list)
    ax = df.plot.bar(rot=90,
                     color={'Machines': 'green', 'Processing Time': 'blue', 'Starving Time': 'red', 'Jobs': 'orange'})
    plt.show()
    """


calculate_starving(amount_m=amount_m, starting_time_ij=starting_time_ij,
                   completion_time_ij=completion_time_ij,
                   starting_time_mw=starting_time_mw, completion_time_mw=completion_time_mw,
                   y_jirm=y_jirm, y_mwr=y_mwr, maintenance_time=maintenance_time, i_j=i_j, study_name=study_name,
                   z_mu=z_mu, makespan=makespan)
