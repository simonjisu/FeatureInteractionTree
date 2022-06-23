
import numpy as np
import shap
import itertools
import time

from .functions import *
from .utils import flatten
from .fitter import PolyFitter
from .dataset import Dataset

from typing import Dict, Any
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from anytree import Node
from .interaction_tree import ShapInteractionTree

class TreeBuilder():
    def __init__(self, model, dataset, original_score: None|float=None):
        """Interaction Tree Builder"""
        self.explainer = shap.TreeExplainer(model)
        self.dataset = dataset
        self.original_score = original_score
        self.score_methods = {
            'base': g_base,
            'abs': g_abs,
            'abs_interaction': g_abs_only_interaction,
            'ratio': g_ratio,
        }
        self.polyfitter = PolyFitter(self.dataset.task_type, self.dataset.data, self.original_score)

    def reset_tree(self, verbose):
        self.root = None
        self.infos = defaultdict(dict)
        self.logger = tqdm(desc='Building Tree')
        self.verbose = verbose

    def shap_interaction_values(self, group_id: None | int=None):
        start = time.time()
        print('Getting Interaction Values via SHAP package, might take a while...')
        if group_id is None:
            # run all data
            # self.polyfitter = PolyFitter(self.dataset.task_type, self.dataset.data, self.original_score)
            print(f'Processing: # of data = {len(self.dataset.data["X_train"])}, # of features = {len(self.dataset.feature_names)}')
            shap_interactions = self.explainer.shap_interaction_values(self.dataset.data['X_train'])
        else:
            # run seperate group
            # polifitter should mimic subset of group data to build tree
            data = self.dataset[group_id]
            # self.polyfitter = PolyFitter(self.dataset.task_type, data, self.original_score)

            print(f'Processing: # of data = {len(data["X_train"])}, # of features = {len(self.dataset.feature_names)}')
            shap_interactions = self.explainer.shap_interaction_values(data['X_train'])
        end = time.time()
        time_cost = end - start
        mins = int(time_cost//60)
        secs = time_cost - mins*60
        print(f'Cost time: {mins:d} mins {secs:.2f} secs')
        return shap_interactions
        
    def build(
            self, 
            score_method: str, 
            shap_interactions: np.ndarray, 
            n_select: int=5, 
            r_filter: float = 0.5,  # filter ratio
            max_iter: int | None=None,
            rt_only_best: bool=True,
            verbose: bool=False
        ):
        self.reset_tree(verbose)
        k = 0
        g_fn = self.score_methods[score_method]
        # feature settings
        self.feature_names = list(map(str, np.arange(shap_interactions.shape[-1])))

        if shap_interactions.ndim == 3:
            # ndim == 3 case: global tree
            build_global = True
        elif shap_interactions.ndim == 2:
            # ndim == 2 case: single tree
            build_global = False
        else:
            raise ValueError('number of dimension of `shap_interactions` should be 2 or 3')

        siv_scores = g_fn(shap_interactions, build_global)
        
        # algorithm will run by number of max features: 
        # can always stop by max_iter, if set as None will automatically set to n_features - 1
        self.n_features = siv_scores.shape[1]
        self.max_iter = self.n_features if max_iter is None else min(max_iter, self.n_features)
        # logger
        self.logger.total = self.max_iter
        self.logger.refresh()
        
        # initialize: main effects
        r_diag, c_diag = np.diag_indices(len(self.feature_names))
        main_effect = siv_scores[r_diag, c_diag]
        self.infos[k]['nodes'] = [dict()]
        self.infos[k]['done'] = [set()]
        for i, name in enumerate(self.feature_names):
            self.infos[k]['nodes'][0][i] = Node(
                name=name, parent=None, score=main_effect[i], interaction=0.0, k=0, drop=0.0
            )

        # TODO: Select Random Nodes to start? or Start from largest Interactions? or Main Effects?
        # try to filter start nodes with larger main effect, since they have larger impact
        nodes_to_run = [(key, n.score) for key, n in self.infos[k]['nodes'][0].items() if key not in self.infos[k]['done'][0]]
        sorted_nodes_to_run = sorted(nodes_to_run, key=lambda x: x[1], reverse=True)
        filtered_nodes_to_run = list(map(lambda x: x[0], sorted_nodes_to_run[:int(len(sorted_nodes_to_run)*r_filter)]))
        #
        self.infos[k]['nodes_to_run'] = [filtered_nodes_to_run]
        self.infos[k]['performance'] = self.polyfitter.original_score
        self._build(siv_scores, n_select, r_filter, k+1)
        self.logger.close()

        if rt_only_best:
            return self._get_best_tree()
        else:
            return self._get_all_trees()
        
    def _build(self, siv_scores, n_select, r_filter, k):
        self.logger.update(1)
        prev_nodes_to_run = deepcopy(self.infos[k-1]['nodes_to_run'])
        prev_nodes = deepcopy(self.infos[k-1]['nodes'])
        prev_dones = deepcopy(self.infos[k-1]['done'])
        
        if (not prev_nodes_to_run) or (k == self.max_iter):
            return 
        else:
            if self.verbose:
                print(f'Current Step: {k}: # to run {len(prev_nodes_to_run)*len(prev_nodes_to_run[0])}')
            if self.infos.get(k) is None:
                self.infos[k]['nodes'] = []
                self.infos[k]['done'] = []
                self.infos[k]['nodes_to_run'] = []
                self.infos[k]['performance'] = []
                
            while prev_nodes_to_run:
                nodes_to_run = prev_nodes_to_run.pop(0)
                nodes = prev_nodes.pop(0)
                done = prev_dones.pop(0)
                existed_cmbs = list(filter(lambda x: isinstance(x, tuple), nodes.keys()))
                scores = self.get_scores(siv_scores, nodes_to_run)
                filtered_keys = self.filter_scores(scores, r_filter, existed_cmbs)
                if self.verbose:
                    print(f'Number of filtered keys: {len(filtered_keys)}')
                for cmbs in tqdm(filtered_keys, total=len(filtered_keys)):
                    new_nodes = deepcopy(nodes)
                    new_done = deepcopy(done)

                    value, interaction = self.get_value_and_interaction(siv_scores, cmbs)
                    
                    feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(cmbs)])
                    
                    children = [new_nodes[c] for c in cmbs]
                    new_nodes[cmbs] = Node(
                        name=feature_name, 
                        score=value, 
                        interaction=interaction, 
                        children=children, 
                        k=k,
                        drop=0.0
                    )
                    performance = self.polyfitter.fit_selected(new_nodes.keys())#, self.feature_names)
                    performance_drop = self.infos[0]['performance'] - performance
                    new_nodes[cmbs].drop = performance_drop

                    self.infos[k]['performance'].append(performance_drop)
                    self.infos[k]['nodes'].append(new_nodes)

                    # add impossibles cmbs
                    for c in cmbs:
                        new_done.add(c)

                    self.infos[k]['done'].append(new_done)
                    new_nodes_to_run = [k for k in new_nodes.keys() if k not in new_done]
                    self.infos[k]['nodes_to_run'].append(new_nodes_to_run)

            # eval
            # for nodes in self.infos[k]['nodes']:
            #     performance = self.polyfitter.fit_selected(nodes.keys(), self.feature_names)
            #     performance_drop = self.infos[0]['performance'] - performance
            #     self.infos[k]['performance'].append(performance_drop)

            sorted_idx = np.argsort(self.infos[k]['performance'])#[::-1]
            # TODO: GA way insert random mutation
            self.infos[k]['nodes_to_run'] = np.array(self.infos[k]['nodes_to_run'], dtype=object)[sorted_idx[:n_select]].tolist()
            self.infos[k]['nodes'] = np.array(self.infos[k]['nodes'], dtype=object)[sorted_idx[:n_select]].tolist()
            self.infos[k]['done'] = np.array(self.infos[k]['done'], dtype=object)[sorted_idx[:n_select]].tolist()
            self.infos[k]['performance'] = np.array(self.infos[k]['performance'])[sorted_idx[:n_select]].tolist()
            
            if self.verbose:
                print(f'Scores: {[round(x, 4) for x in self.infos[k]["performance"]]}')
            return self._build(siv_scores, n_select, r_filter, k+1)
    
    def filter_scores(self, scores, r_filter, existed_cmbs=None):
        if len(scores) == 1:
            return list(scores.keys())
        filtered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if existed_cmbs is not None:
            filtered = [(key, value) for key, value in filtered if key not in existed_cmbs]
        n_filterd = int(r_filter * len(filtered))
        return list(map(lambda x: x[0], filtered[:n_filterd]))

    def get_value_and_interaction(self, siv_scores, cmbs):
        r_l, c_l = np.tril_indices(self.n_features, -1)
        cmbs_flattend = list(flatten(cmbs))
        cmbs_idx = np.arange(len(r_l))[np.isin(r_l, cmbs_flattend) & np.isin(c_l, cmbs_flattend)]

        r, c = list(zip(*itertools.product(flatten(cmbs), flatten(cmbs))))
        value = siv_scores[r, c].sum()
        interaction = siv_scores[r_l, c_l][cmbs_idx].sum()
        return value, interaction

    def get_scores(self, siv_scores, nodes_to_run):
        scores = {}
        for cs in itertools.combinations(nodes_to_run, 2):
            if cs not in scores.keys():
                r, c = list(zip(*itertools.product(flatten(cs), flatten(cs))))
                scores[cs] = siv_scores[r, c].sum()
        return scores

    def _extract_root(self, node_record):
        root = list(node_record.values())[-1]
        return root

    def _get_all_trees(self):
        last_records = list(self.infos.values())[-1]
        trees = []
        for node_record in last_records['nodes']:
            root = self._extract_root(node_record)
            trees.append(ShapInteractionTree(root))
        return trees

    def _get_best_tree(self):
        last_records = list(self.infos.values())[-1]
        best_idx = np.argmax(last_records['performance'])
        node_record = last_records['nodes'][best_idx]
        root = self._extract_root(node_record)
        return ShapInteractionTree(root)