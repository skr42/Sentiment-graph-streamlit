import glob
import streamlit as st
import plotly.express as px

import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer

# Set the NLTK data directory
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
nltk.data.path.append(nltk_data_dir)

# Download the VADER lexicon data
nltk.download('vader_lexicon', download_dir=nltk_data_dir)

filepaths = sorted(glob.glob("diary/*.txt"))

analyzer = SentimentIntensityAnalyzer()

negativity = []
positivity = []
for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    scores = analyzer.polarity_scores(content)
    positivity.append(scores["pos"])
    negativity.append(scores["neg"])

dates = [name.strip(".txt").strip("diary/") for name in filepaths]

st.title("Diary Tone")
st.subheader("Positivity")
pos_figure = px.line(x=dates, y=positivity,
                     labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(pos_figure)

st.subheader("Negativity")
neg_figure = px.line(x=dates, y=negativity,
                     labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(neg_figure)
