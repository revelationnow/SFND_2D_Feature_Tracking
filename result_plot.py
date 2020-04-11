#!/usr/bin/env python3

import pandas as pd
import plotly.express as px
import plotly

a = pd.read_csv('./build/results.csv')
fig = px.box(a, x="Detector",color="Descriptor", y="num_keypoints", points="all")
fig.write_html("keypoints.html")
fig = px.box(a, x="Detector",color="Descriptor", y="num_matches", points="all")
fig.write_html("matches.html")
fig = px.box(a, x="Detector",color="Descriptor", y="time_take", points="all")
fig.write_html("time_taken.html")
