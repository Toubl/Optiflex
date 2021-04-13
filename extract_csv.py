import pandas as pd
import numpy as np


def extract_parameter(input_path, variants, amount):

    df = pd.read_csv(input_path, sep="\t", engine="python", header=None)
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
        "AFO160",
        "AFO300",
        "AFO170",
        "AFO190",
        "AFO220",
    ]
    max_machines = 7
    output = np.zeros((len(variants), len(afo_list), max_machines * len(afo_list)))
    n = 0
    for name in variants:
        line = 0
        filtered_table = df[df[4].str.match(name)]
        for afo in filtered_table[0]:
            position = np.where(np.asarray(afo_list) == afo)
            if position is []:
                afo_list.append(afo)
                position = len(afo_list) - 1
            machine = (
                int((filtered_table.iloc[line, 2])[0:3])
                - 1
                + (position[0] * max_machines)
            )
            output[n, position, machine] = filtered_table.iloc[line, 6]
            line += 1
        n += 1

    p_jim = np.zeros((np.sum(amount), len(afo_list), max_machines * len(afo_list)))

    job = 0
    var = 0
    for number in amount:
        for n in range(number):
            p_jim[job + n, :, :] = output[var, :, :]
        job += number
        var += 1

    return p_jim.astype(int)


processing_time_path = "/Users/q517174/PycharmProjects/Optiflex/parameter/Taktzeit.csv"
variants_of_interest = ["B37 C15 TUE1", "B48 B20 TUE1", "B38 A15 TUE1"]
amount_of_variants = [3, 3, 3]
processing_time_jim = extract_parameter(
    processing_time_path, variants_of_interest, amount_of_variants
)
