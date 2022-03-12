#!/usr/bin/env python3

import sys
import pandas as pd #Librería para el manejo de datos en Python. Permite realizar vi
import numpy as np #Librería para computación numérica en Python.
import plotly.graph_objs as go
import plotly.express as px
import plotly.subplots as ps
import plotly.figure_factory as ff


def main():
    basePath = sys.argv[1]

    values = pd.read_csv(basePath + "Values/part-00000", header=None) \
            .rename(columns={0:"values"}, errors="raise")
    ranges = pd.read_csv(basePath + "Ranges/part-00000", header=None) \
            .rename(columns={0:"ranges"}, errors="raise")

    arr = ranges.values.tolist()
    arr = [x[0] for x in arr]
    tmp = []
    for i in range(len(arr)-1):
            tmp.append( str([arr[i], arr[i+1]]) )

    ranges2 = pd.DataFrame(tmp).rename(columns={0:"ranges"}, errors="raise")


    conc = pd.concat([values, ranges2], axis=1)

    fig = px.bar(conc, x="ranges", y="values")
    fig.show()


if __name__ == "__main__":
    main()

