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

    values = pd.read_csv(basePath + "Values.txt", header=None) \
            .rename(columns={0:"values"}, errors="raise")
    tags = pd.read_csv(basePath + "Tags.txt", header=None) \
            .rename(columns={0:"tags"}, errors="raise")

    values2 = values.values.tolist()
    tags2 = tags.values.tolist()[0]

    fig = px.imshow(values, x=tags2, y=tags2)
    fig.show()


if __name__ == "__main__":
    main()

