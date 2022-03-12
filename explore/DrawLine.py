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
    tags = pd.read_csv(basePath + "Tags/part-00000", header=None) \
            .rename(columns={0:"ranges"}, errors="raise")

    conc = pd.concat([values, tags], axis=1)

    fig = px.line(conc, x="ranges", y="values")
    fig.show()


if __name__ == "__main__":
    main()

