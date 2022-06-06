# streamlit app for tree demo

import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

st.title('Feature Interaction Tree with SHAP Interaction Values')
st.write('''
## Purpose and Motivation

Sometimes it is not straight-forward to understand clearly for the receivers of a SHAP explanation (Need to know what the graph means).
When features interact with each other in a prediction model, the prediction cannot be expressed as the sum of the feature effects, 
because the effect of one feature depends on the value of the other feature. When there are numerous features, it is hard to see 
overall pattern between features. We want to explore the interaction effect between features with SHAP interaction values. 

## What is SHAP Interaction Values?

The interaction effect is an additional feature effect when two(or more) features combined together after subtracting 
each features' individual effect(usually call it main effect). Usually, the combined main effects are bigger than the 
interaction effect. [The Shapley interaction index](https://arxiv.org/abs/1902.05622) is defined as below from game theory.

$$\phi_{i,j}=\sum_{S\subseteq M \setminus\{i,j\}} \\dfrac{ \\vert S\\vert !( \\vert M \\vert - \\vert S \\vert -2)!}{2( \\vert M \\vert -1)!} \delta_{ij}(S)$$

where $i\\neq j$ and $\delta_{ij}(S)=\hat{f}_x(S\cup\{i,j\})-\hat{f}_x(S\cup\{i\})-\hat{f}_x(S\cup\{j\})+\hat{f}_x(S)$

## Why using Tree?

* Tree is accepted as one of the most commonly used explaining method. "Tree-based methods are used widely in industry."
* With tree architecture, people can easily understand how model behaves.
* Hierarchical structures such as trees are useful in expressing levels of both "when inputs are perturbed" and "when models are perturbed"
* Furthermore, they are useful in displaying the difference of importances.

## Algorithms

The algorithm using recursive function to build a tree with hyperpameters of `Method` and `top N`. 

### The meaning of Method 

Method is related to how do we deal with the interaction values. We try to make the tree algorithm work on both 
local and global, since user might want to know the interaction effects globally. These methods are also point of views 
to process the shap interaction values. After process them to scores, we use these scores to decide how to 
build a feature tree.

$$scores = g(\\text{SHAP interaction values})$$

* `base`: Average the values at global mode. This means we consider both direction and magnitude of interaction effects and main effects.
* `abs`: First calculate the absolute values and then average them at global mode. This means we only consider the magnitude of interaction effects and main effects. 
* `abs_interaction`: Calculate the absolute values and then average them at global mode. Also, replace all main effects to zeros. This means we only consider the magnitude of interaction effects. 
* `ratio`: Calculate the absolute value, average them at global mode. Then, devide the summation of the combination of main effects, and fill all main effects to zeros. This means we consider the relative size of interaction effects.

### The meaning of Top N

The algorithm build the tree by selecting Top N Scores. At each iteration of calling, it search $C_M^2$ combinations of scores 
to build a tree. For example, if top N equals to 1, it means we consider the highest interaction impact to combine the features. 
We did not randomly choose nodes to build a tree since that will not fit to explaination purpose. 

Also, the calculatation cost in the shap interaction value for each example will cost $O(TMLD^2)$ in TreeSHAP. The tree building process 
will cost only on sorting $O(\\frac{M^2-M}{2} \log \\frac{M^2-M}{2}) \\approx O(M^2 \\log M) $ for each depth of tree.
* $T$: the number of trees
* $L$: the maximum number of leaves in any tree
* $D$: the depth of tree, $\log L$
* $M$: the muber of features

''')

st.write('''
## Example

* `Dataset`: [California Housing Price Prediction](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
* `Model`: XGBoost with learning rate = 0.3, max depth = 8, number of rounds = 200
* `Performance`: 
    * number of train data: 18576
    * number of test data: 2064
    * R2 square on Test Set: 0.8453
''')

feature_names, shap_values, shap_interaction_values = get_values()
tree_builder = TreeBuilder()
magnitude = False 

with st.expander('Some Plots about SHAP Interaction Values'):
    df_shap = pd.DataFrame(
        shap_values.values.mean(0)[:, np.newaxis].round(4).T, 
        columns=feature_names, 
        index=['Average SHAP values']
    )
    st.write(df_shap.style.format("{:.4}"))
    st.write(f'Sum of all averaged global shap values: {shap_values.values.mean(0).sum():.4f}')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.heatmap(
        data=pd.DataFrame(shap_interaction_values.mean(0), index=feature_names, columns=feature_names),
        cmap='coolwarm', annot=True, fmt='.4f', ax=ax
    )
    ax.set_title('Average on SHAP Interaction Values')
    st.pyplot(fig)

st.write('## Step by step building a feature tree')

col1, col2 = st.columns(2)
with col1:
    s = st.selectbox(
        label='Method of building trees', 
        options=list(tree_builder.score_methods.keys()),
        index=1
    )
    
with col2:
    top_n = st.selectbox(
        label='Top N', 
        options=list(range(1, 4)),
        index=1
    )

nodes = tree_builder.build(
    s, 
    shap_interactions=shap_interaction_values, 
    feature_names=feature_names, 
    top_n=top_n, 
    magnitude=magnitude
)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
sns.heatmap(
    data=pd.DataFrame(tree_builder.siv_scores, index=feature_names, columns=feature_names),
    cmap='coolwarm', annot=True, fmt='.4f', ax=ax
)
ax.set_title('Score Matrix')
st.pyplot(fig)

step = st.slider(label='Step:', min_value=0, max_value=len(tree_builder._iterations)-1, value=0)
img = tree_builder.show_step_by_step(step)
st.image(img)