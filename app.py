# streamlit app for tree demo

from collections import defaultdict 
import shap
import os
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import pickle
from pathlib import Path
from ds_desc import DESC
import xgboost as xgb
from fge import ShapInteractionTree, Dataset

EXP_DIR = 'onehot_intercept'  # onehot_nointercept
st.set_page_config(layout="wide")

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
num_rounds = cache[ds_name]['model_args']['num_rounds']
origin_score = trees_dicts['origin_score']
linear_score = trees_dicts['linear_score']

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
st_shap(shap.plots.bar(shap_values, max_display=20), height=400, width=700)
st.write(DESC[ds_name])

st.write('# Experiment Plots')

st.write(f"""Model: 
* Number of boost round = {num_rounds}
* XGBoost model performance = {origin_score:.4f}
* Linear model performance = {linear_score:.4f}
""")

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
