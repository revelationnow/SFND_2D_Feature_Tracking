#!/usr/bin/env python3

import pip

def install(package):
    pip.main(['install', package])

if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        install('pandas')
        import pandas as pd

    try:
        import plotly
        import plotly.express as px
    except ImportError:
        install('plotly')
        import plotly
        import plotly.express as px

    try:
        import psutil
    except ImportError:
        install('psutil')
        import psutil

a = pd.read_csv('./build/results.csv')
fig = px.box(a, x="Detector",y="num_keypoints", points="all")
fig.write_html("keypoints.html")
fig.write_image("keypoints.svg")
fig = px.box(a, x="Detector",color="Descriptor", y="num_matches", points="all")
fig.write_html("matches.html")
fig.write_image("matches.svg")
fig = px.box(a, x="Detector",color="Descriptor", y="keyp_by_matches", points="all")
fig.write_html("keyp_by_matches.html")
fig.write_image("keyp_by_matches.svg")
fig = px.box(a, x="Detector",color="Descriptor", y="time_take", points="all")
fig.write_html("time_taken.html")
fig.write_image("time_taken.svg")
