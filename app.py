# streamlit app for tree demo

import shap
import numpy as np
import pandas as pd
import streamlit as st
import pickle

from pathlib import Path

from fge.tree_builder import TreeBuilder

st.set_page_config(layout="wide")

@st.cache
def get_values():
    cache_path = Path('.').absolute() / 'cache'
    with (cache_path / 'CAmodel.pickle').open('rb') as file:
        results = pickle.load(file)

    model = results['model']
    # X_train, y_train = results['train']
    X_test, _ = results['test']

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    feature_names = X_test.columns
    return feature_names, shap_values, shap_interaction_values

feature_names, shap_values, shap_interaction_values = get_values()
tree_builder = TreeBuilder()
magnitude = False 
top_n = 2

st.title('Demo on Tree Building')
df = pd.DataFrame(shap_values.values.mean(0)[:, np.newaxis].round(4), index=feature_names, columns=['Average SHAP values'])
st.write(df.style.format("{:.4}"))
st.write(f'Sum of all averaged global shap values: {shap_values.values.mean(0).sum():.4f}')

s = st.selectbox(
    label='Method of building trees', 
    options=list(tree_builder.score_methods.keys())
)

nodes = tree_builder.build(
    s, 
    shap_interactions=shap_interaction_values, 
    feature_names=feature_names, 
    top_n=top_n, 
    magnitude=magnitude
)

step = st.slider(label='Step:', min_value=0, max_value=len(tree_builder._iterations)-1, value=0)
img = tree_builder.show_step_by_step(step)
st.image(img)