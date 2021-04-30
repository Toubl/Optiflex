import pandas as pd
import numpy as np
import copy


def extract_parameter(input_path, variants, amount, maintenance=None):

    df = pd.read_excel(input_path, skiprows=0, header=0, sheet_name='Sheet2')
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
    max_machines = 7
    output = np.zeros((len(variants), len(afo_list), max_machines * len(afo_list)))
    n = 0
    for name in variants:
        line = 0
        var = df[name]
        for afo in df['AFO']:
            position = np.where(np.asarray(afo_list) == afo)
            if position is []:
                afo_list.append(afo)
                position = len(afo_list) - 1
            machine = (
                int((df.iloc[line, 1]))
                - 1
                + (position[0] * max_machines)
            )
            output[n, position, machine] = var[line]
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

    d_mw = None
    if maintenance is not None:
        df_maint = pd.read_excel(input_path, skiprows=0, header=0, sheet_name='Maintenance')
        maintenance_activities = len(df_maint.columns) - 2
        d_mw = np.zeros((max_machines * len(afo_list), maintenance_activities))
        line = 0
        for afo in df_maint['AFO']:
            position = np.where(np.asarray(afo_list) == afo)
            if position is []:
                afo_list.append(afo)
                position = len(afo_list) - 1
            machine = (
                int((df.iloc[line, 1]))
                - 1
                + (position[0] * max_machines)
            )
            d_mw[machine, :] = df_maint.iloc[line, 2:]
            line += 1
        n += 1

    return p_jim.astype(int), d_mw


def generate_new_list(input_path):
    df = pd.read_excel(input_path, skiprows=0, header=0)
    new_df = copy.deepcopy(df)
    new_df.drop_duplicates(subset=['AFO', 'Maschine'], keep='first', inplace=True)
    new_df = new_df.iloc[:, [0, 2]]
    new_df = new_df.reset_index(drop=True)
    line = 0
    for afo in df['AFO']:
        machine = df.iloc[line, 2]
        product = df.iloc[line, 6]
        time = df.iloc[line, 8]
        if product not in new_df.columns:
            new_df[product] = 0

        index = list((new_df.loc[(new_df.AFO == afo) & (new_df.Maschine == machine)]).index)
        index_col = new_df.columns.get_loc(product)
        print(index, index_col)
        new_df.iloc[index[0], index_col] = time

        line += 1
    return new_df

processing_time_path = "parameter/Takzeit_overview.xlsx"
variants_of_interest = ["B37 D", "B37 C15 TUE1", "B48 B20 TUE1", "B38 A15 TUE1"]
amount_of_variants = [3, 3, 3, 3]
# new_df = generate_new_list(processing_time_path)
# new_df.to_excel(r'parameter/Takzeit_overview.xlsx', index=False)

processing_time_jim = extract_parameter(
    processing_time_path, variants_of_interest, amount_of_variants, maintenance=True
)

b = 0
