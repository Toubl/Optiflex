import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

processing_time_path = "parameter/Takzeit_overview.xlsx"

df = pd.read_excel(processing_time_path, skiprows=0, header=0, sheet_name="Sheet2")
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

product_list = ["B37 D", "B37 C15 TUE1", "B48 B20 TUE1", "B38 A15 TUE1"]
scale = 1 / 100
xoffset = 3
rad_min = 0.8
x = 0
mg = 0

with open("pie_chart_file.txt", "w") as text_file:
    text_file.write('\\begin{tikzpicture}' + "\n")
    for afo in afo_list:
        radius = []
        values = []  # [[] for i in range(len(product_list))]
        count = 0
        for product in product_list:

            # text_file.write(afo + "\n")
            stuff = (df[df.AFO == afo][product]).tolist()
            stuff = [i for i in stuff if i != 0]

            if len(stuff) > 0:
                radius.append(sum(stuff) / (len(stuff) ** 2) * scale)
                val = ""
                for t in stuff:
                    val += str(t) + "/,"
                values.append(val[:-1])
            else:
                radius.append(0)
                values.append('')

        # determine positions
        if afo == 'AFO005':
            ypos = 0
            line_pos = max(max(radius) * 1.1, rad_min/2)
        else:
            rad_sum = [x + y for x, y in zip(radius, old_radius)]
            ypos += - max(max(rad_sum) * 1.05, rad_min)
            line_pos = ypos + max((max(radius) * 1.05), 0)

        old_radius = radius

        for i in range(len(product_list)):
            werte = str(values[i])
            xpos = i * xoffset
            output_string = (
                    "\pie[pos={%s,%s},radius={%s}, sum=auto,before number=\phantom,after number=]{%s}\n"
                    % (xpos, ypos, radius[i], werte)
            )

            text_file.write(output_string)
        # draw vertical line
        text_file.write('\\draw[] (-3,{}) to[|-|] (10,{});'.format(line_pos, line_pos) + '\n')
        text_file.write('\\node[align=left, text width=4cm] at (-1,{}) {{{}}};'.format(ypos, ('MG' + str(mg))) + '\n')
        mg += 1

    text_file.write('\\node[anchor=mid] at (0,1) {Product 1};' + '\n')
    text_file.write('\\node[anchor=mid] at (3,1) {Product 2};' + '\n')
    text_file.write('\\node[anchor=mid] at (6,1) {Product 3};' + '\n')
    text_file.write('\\node[anchor=mid] at (9,1) {Product 4};' + '\n')
    text_file.write('\\draw[very thick] (1.5,1.5) to[|-|] (1.5,{});'.format(ypos-0.3) + '\n')
    text_file.write('\\draw[very thick] (4.5,1.5) to[|-|] (4.5,{});'.format(ypos-0.3) + '\n')
    text_file.write('\\draw[very thick] (7.5,1.5) to[|-|] (7.5,{});'.format(ypos-0.3) + '\n')
    text_file.write('\\draw[very thick] (-1.5,1.5) to[|-|] (-1.5,{});'.format(ypos - 0.3) + '\n')
    text_file.write('\\end{tikzpicture}' + "\n")
# output_string = '\pie{[pos={0,0},radius={%s}, color={%s}]{%s}}' % (radius, color, values)
