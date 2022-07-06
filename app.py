# streamlit app for tree demo

from collections import defaultdict 
import shap
import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import pickle
import seaborn as sns
from pathlib import Path
from ds_desc import DESC
import xgboost as xgb
from fge import ShapInteractionTree, Dataset

EXP_DIR = 'onehot_intercept'  # onehot_nointercept
st.set_page_config(layout="wide")

# st.title('Feature Interaction Tree with SHAP Interaction Values')
# st.write('''
# ## Purpose and Motivation

# Sometimes it is not straight-forward to understand clearly for the receivers of a SHAP explanation (Need to know what the graph means).
# When features interact with each other in a prediction model, the prediction cannot be expressed as the sum of the feature effects, 
# because the effect of one feature depends on the value of the other feature. When there are numerous features, it is hard to see 
# overall pattern between features. We want to explore the interaction effect between features with SHAP interaction values. 

# ## What is SHAP Interaction Values?

# The interaction effect is an additional feature effect when two(or more) features combined together after subtracting 
# each features' individual effect(usually call it main effect). Usually, the combined main effects are bigger than the 
# interaction effect. [The Shapley interaction index](https://arxiv.org/abs/1902.05622) is defined as below from game theory.

# $$\phi_{i,j}=\sum_{S\subseteq M \setminus\{i,j\}} \\dfrac{ \\vert S\\vert !( \\vert M \\vert - \\vert S \\vert -2)!}{2( \\vert M \\vert -1)!} \delta_{ij}(S)$$

# where $i\\neq j$ and $\delta_{ij}(S)=\hat{f}_x(S\cup\{i,j\})-\hat{f}_x(S\cup\{i\})-\hat{f}_x(S\cup\{j\})+\hat{f}_x(S)$

# ## Why using Tree?

# * Tree is accepted as one of the most commonly used explaining method. "Tree-based methods are used widely in industry."
# * With tree architecture, people can easily understand how model behaves.
# * Hierarchical structures such as trees are useful in expressing levels of both "when inputs are perturbed" and "when models are perturbed"
# * Furthermore, they are useful in displaying the difference of importances.

# ## Algorithms

# The algorithm using recursive function to build a tree with hyperpameters of `Method` and `top N`. 

# ### The meaning of Method 

# Method is related to how do we deal with the interaction values. We try to make the tree algorithm work on both 
# local and global, since user might want to know the interaction effects globally. These methods are also point of views 
# to process the shap interaction values. After process them to scores, we use these scores to decide how to 
# build a feature tree.

# $$scores = g(\\text{SHAP interaction values})$$

# * `base`: Average the values at global mode. This means we consider both direction and magnitude of interaction effects and main effects.
# * `abs`: First calculate the absolute values and then average them at global mode. This means we only consider the magnitude of interaction effects and main effects. 
# * `abs_interaction`: Calculate the absolute values and then average them at global mode. Also, replace all main effects to zeros. This means we only consider the magnitude of interaction effects. 
# * `ratio`: Calculate the absolute value, average them at global mode. Then, devide the summation of the combination of main effects, and fill all main effects to zeros. This means we consider the relative size of interaction effects.

# ### The meaning of Top N

# The algorithm build the tree by selecting Top N Scores. At each iteration of calling, it search $C_M^2$ combinations of scores 
# to build a tree. For example, if top N equals to 1, it means we consider the highest interaction impact to combine the features. 
# We did not randomly choose nodes to build a tree since that will not fit to explaination purpose. 

# Also, the calculatation cost in the shap interaction value for each example will cost $O(TMLD^2)$ in TreeSHAP. The tree building process 
# will cost only on sorting $O(\\frac{M^2-M}{2} \log \\frac{M^2-M}{2}) \\approx O(M^2 \\log M) $ for each depth of tree.
# * $T$: the number of trees
# * $L$: the maximum number of leaves in any tree
# * $D$: the depth of tree, $\log L$
# * $M$: the muber of features

# ''')

# st.write('''
# ## Example

# * `Dataset`: [California Housing Price Prediction](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
# * `Model`: XGBoost with learning rate = 0.3, max depth = 8, number of rounds = 200
# * `Performance`: 
#     * number of train data: 18576
#     * number of test data: 2064
#     * R2 square on Test Set: 0.8453
# ''')

@st.cache(hash_funcs={shap.Explainer: hash, ShapInteractionTree: hash, xgb.Booster: hash, Dataset:hash}, suppress_st_warning=True)
def load_cache(cache_path, dataset_names):
    cache = defaultdict()
    system = '_win' if os.name == 'nt' else ''
    for ds_name in dataset_names:
        with (cache_path / f'{ds_name}{system}.pickle').open('rb') as file:
            res = pickle.load(file)
        cache[ds_name] = res
        dataset = cache[ds_name]['dataset']
        explainer = cache[ds_name]['explainer']
        data = dataset.data['X_train']
        sv = explainer.shap_values(data)
        cache[ds_name]['shap_values'] = sv
    return cache

def get_exp_name(sm, nsn, nsg, ntrm, fm):
    args = map(lambda x: str(x), [sm, nsn, nsg, ntrm, fm])
    exp_name = '_'.join(args)
    return exp_name

@st.cache
def get_df_gaps(cache_path):
    # df_gaps
    df_gaps = pd.read_csv(cache_path / 'all_results.csv', encoding='utf-8')
    df_gaps['method'] = df_gaps['exp_name'].str.extract('(abs\_interaction|ratio|abs)')
    df_gaps['exps'] = df_gaps.loc[:, 'exp_name'].values.copy()
    for method in df_gaps['method'].unique():
        cond = df_gaps['method'] == method
        df_gaps.loc[cond, 'exps'] = df_gaps.loc[cond, 'exps'].str.slice(start=len(method))
    # df_gaps['exp_name'] = df_gaps['exp_name'].str.slice(start=1)
    rename_cols = {0: 'n_select_scores', 1: 'n_select_gap', 2: 'nodes_to_run_method', 3: 'filter_method'}
    df_gaps = pd.concat([df_gaps, df_gaps['exps'].str.slice(start=1).str.split('_', expand=True).rename(columns=rename_cols)], axis=1)
    df_gaps.drop(columns=['exps'], inplace=True)
    df_gaps['n_select_scores'] = df_gaps['n_select_scores'].astype(np.int32)
    df_gaps['n_select_gap'] = df_gaps['n_select_gap'].astype(np.int32)
    return df_gaps


dataset_names = ['titanic', 'adult', 'boston', 'california']
cache_path = Path('.').resolve() / 'cache' / EXP_DIR 
cache = load_cache(cache_path, dataset_names)
df_gaps = get_df_gaps(cache_path)

score_method_list = ['abs', 'abs_interaction', 'ratio']
n_select_scores_list = [5, 10]
n_select_gap_list = [5, 10]
nodes_to_run_method_list = ['random', 'sort', 'full']
filter_method_list = ['random', 'sort', 'prob']


# individual exp
with st.sidebar:
    ds_name = st.selectbox(label='Dataset Name', options=dataset_names, index=0)
    score_method = st.selectbox(label='Score Method', options=score_method_list, index=0)
    n_select_scores = st.selectbox(label='Maximum Numer of selecting candidates', options=n_select_scores_list, index=0)
    n_select_gap = st.selectbox(label='Maximum Numer of keeping best gap', options=n_select_gap_list, index=0)
    nodes_to_run_method = st.selectbox(label='Way to select nodes', options=nodes_to_run_method_list, index=0)
    filter_method = st.selectbox(label='Way to filter nodes', options=filter_method_list, index=0)

    st.write("""
    Notations:

    - `K` = Maximum Numer of selecting candidates
    - `scores` = Dictionary of 
        - Key: Combination of features
        - Value: SIV Score of Combination of features
    
    Arguments: 

    - `random`: random select by `K` nodes
    - `sort`: sorting `scores` by its values, then select top `K` nodes
    - `full`: select all avaliable nodes for candidates
    - `prob`: select nodes from the probability of candidates which can be calculated from `scores` (only used in filter)

    """)

exp_name = get_exp_name(score_method, n_select_scores, n_select_gap, nodes_to_run_method, filter_method)
trees_dicts = cache[ds_name]['trees'][exp_name]
#origin_score = trees_dicts['origin_score']
#linear_score = trees_dicts['linear_score']
tree = trees_dicts['t'][-1]

dataset = cache[ds_name]['dataset']
siv = cache[ds_name]['siv']
sv = cache[ds_name]['shap_values']
explainer = cache[ds_name]['explainer']
features = dataset.data['X_train']

shap_values = shap.Explanation(
    sv, 
    base_values=explainer.expected_value,
    data=features,
    feature_names=dataset.feature_names
)

st.write("## Tree")
tree_img = tree.show_tree(feature_names=dataset.feature_names)
st.image(tree_img)


st.write("## SHAP Plots")
n = 1000
shap.initjs()
st_shap(shap.summary_plot(shap_values), height=400, width=700)
# st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000], features.iloc[:1000]))
st_shap(shap.plots.bar(shap_values, max_display=20), height=400, width=700)
st.write(DESC[ds_name])

st.write('# Experiment Plots')

op_sm = st.multiselect('Score Method', options=score_method_list)
op_nss = st.multiselect('Maximum Numer of selecting candidates', options=n_select_scores_list)
op_nsg = st.multiselect('Maximum Numer of keeping best gap', options=n_select_gap_list)
op_ntrm = st.multiselect('Way to select nodes', options=nodes_to_run_method_list)
op_fm = st.multiselect('Way to filter nodes', options=filter_method_list)

score_method_list = ['abs', 'abs_interaction', 'ratio']
n_select_scores_list = [5, 10]
n_select_gap_list = [5, 10]
nodes_to_run_method_list = ['random', 'sort', 'full']
filter_method_list = ['random', 'sort', 'prob']


df_exps = df_gaps.loc[(df_gaps['dataset'] == ds_name), :]

if op_sm and op_nss and op_nsg and op_ntrm and op_fm:
    cond = df_exps['method'].isin(op_sm) & df_exps['n_select_scores'].isin(op_nss) & df_exps['n_select_gap'].isin(op_nsg) & \
        df_exps['nodes_to_run_method'].isin(op_ntrm) & df_exps['filter_method'].isin(op_fm)
    df_draw = df_exps.loc[df_exps.index[cond], ['step', 'exp_name', 'gaps']]
    pivot_table = df_draw.pivot(index='step', columns='exp_name')['gaps']
    st.write(pivot_table.T)
    st.line_chart(
        pd.DataFrame(pivot_table.values, index=pivot_table.index, columns=pivot_table.columns.values), 
        width=720)
else:
    st.warning('Please select at least one value in each compnent')
    st.stop()
