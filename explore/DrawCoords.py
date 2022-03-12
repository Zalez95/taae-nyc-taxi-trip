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

    df = pd.read_csv(basePath + "/part-00000", header=None) \
            .rename(columns={0:"lat", 1:"lon", 2:"count"}, errors="raise")

    fig = px.scatter_geo(df, lat="lat", lon="lon", size="count")
    fig.update_layout(geo_scope="new york")
    fig.show()

if __name__ == "__main__":
    main()

