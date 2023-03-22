import streamlit as st
import pickle
from powercoder import PowerTagger, TagPrediction
import numpy as np
import pandas as pd

tagger = PowerTagger(model="rfc")

st.write("""
# Welcome to PowerCoder
One stop solution for all coding problems
""")

task = st.sidebar.selectbox("Task", ["Data Structure Tag", "Algorithm Tag"])
classifier = st.sidebar.selectbox("Classifier", ["Random Forest", "XGBoost"])

question = st.text_input("Enter your question: ", "Yada yada yada ya ...")


# task_button = st.button("Submit")

def tag_question(clf, qn):
    tags = {
        0: "graph",
        1: "array",
        2: "string"
    }
    tag_predictions = tagger.predict([qn]).results
    st.write(f"Predicted data structure: {tags[tag_predictions.iloc[0,0]]} with a probability of {tag_predictions.iloc[0, 1]:.3f}")


if st.button("Submit"):
    tag_question(classifier, question)
