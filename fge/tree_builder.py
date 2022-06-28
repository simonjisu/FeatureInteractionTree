import numpy as np
import shap
import itertools
import time
import heapq
import random

from .functions import *
from .utils import flatten
from .fitter import PolyFitter
from .interaction_tree import ShapInteractionTree

from typing import Dict, Any, List
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from anytree import Node

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
        self.verbose = verbose
        if not self.verbose:
            self.logger = tqdm(desc='Building Tree')
        

    def shap_interaction_values(self, group_id: None | int=None):
        start = time.time()
        print('Getting Interaction Values via SHAP package, might take a while...')
        if group_id is None:
            # run all data
            print(f'Processing: # of data = {len(self.dataset.data["X_train"])}, # of features = {len(self.dataset.feature_names)}')
            shap_interactions = self.explainer.shap_interaction_values(self.dataset.data['X_train'])
        else:
            # run seperate group
            # polifitter should mimic subset of group data to build tree
            data = self.dataset[group_id]

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
            n_select_scores: int=5,
            n_select_gain: int=5, 
            max_iter: int | None=None,
            nodes_to_run_method: str | None='random',  # random / sort / full
            filter_method: str = 'random',  # random / sort / prob
            rt_only_best: bool=True,
            verbose: bool=False
        ):
        """
        filter_method(sort) + score_method(only interaction method: abs_interaction, ratio)
        """

        self.reset_tree(verbose)
        # feature settings
        self.feature_names = np.arange(shap_interactions.shape[-1])
        self.nodes_to_run_method = nodes_to_run_method
        self.filter_method = filter_method
        if shap_interactions.ndim == 3:
            # ndim == 3 case: global tree
            build_global = True
        elif shap_interactions.ndim == 2:
            # ndim == 2 case: single tree
            build_global = False
        else:
            raise ValueError('number of dimension of `shap_interactions` should be 2 or 3')
        
        g_fn = self.score_methods[score_method]
        siv_scores = g_fn(shap_interactions, build_global)
        k = 0

        # algorithm will run by number of max features: 
        # can always stop by max_iter, if set as None will automatically set to n_features - 1
        self.n_features = siv_scores.shape[1]
        self.max_iter = self.n_features if max_iter is None else min(max_iter, self.n_features)
        if not self.verbose:
            # logger
            self.logger.total = self.max_iter
            self.logger.refresh()
        
        # initialize: main effects
        r_diag, c_diag = np.diag_indices(len(self.feature_names))
        main_effect = siv_scores[r_diag, c_diag]
        self.infos[k]['nodes'] = [dict()]
        for i, name in enumerate(self.feature_names):
            self.infos[k]['nodes'][0][i] = Node(
                name=str(name), parent=None, score=main_effect[i], interaction=0.0, k=0, gain=None
            )

        nodes_to_run = self.get_nodes_to_run(nodes=self.infos[k]['nodes'][0], n_select_scores=n_select_scores)
        self.infos[k]['nodes_to_run'] = [nodes_to_run]
        self.infos[k]['gain'] = {
            'origin': self.polyfitter.original_score, 
            'min': self.polyfitter.min_score,
        }

        self._build(siv_scores, n_select_scores, n_select_gain, k+1)
        if not self.verbose:
            self.logger.close()
        if rt_only_best:
            return self._get_best_tree()
        else:
            return self._get_all_trees()

    def _build(self, siv_scores, n_select_scores, n_select_gain, k):
        if not self.verbose:
            self.logger.update(1)
        prev_nodes_to_run = deepcopy(self.infos[k-1]['nodes_to_run'])
        prev_nodes = deepcopy(self.infos[k-1]['nodes'])
        
        if (not prev_nodes_to_run) or (k == self.max_iter):
            return 
        else:
            if self.verbose:
                print(f'Current Step: {k}: # to run {len(prev_nodes_to_run)*len(prev_nodes_to_run[0])}')
            if self.infos.get(k) is None:
                self.infos[k]['nodes'] = []
                self.infos[k]['nodes_to_run'] = []
                self.infos[k]['gain'] = []
            
            # deterministic selection
            all_gains = []
            heapq.heapify(all_gains)
            i = 0
            while prev_nodes_to_run:
                nodes_to_run = prev_nodes_to_run.pop(0)
                nodes = prev_nodes.pop(0)
                
                scores = self.get_scores(siv_scores, nodes_to_run)
                filtered_keys = self.filter_scores(scores, n_select_scores)
                if self.verbose:
                    print(f'Number of filtered keys: {len(filtered_keys)}')
                # else:
                #     self.logger.set_postfix_str(f'N keys to run: {len(filtered_keys)}')
                
                for cmbs in filtered_keys:
                    i += 1
                    trials = list(nodes.keys()) + [cmbs]
                    gain = self.polyfitter.get_interaction_gain(trials)
                    if len(all_gains) >= n_select_gain:
                        if all_gains[0][1] > gain:
                            heapq.heappushpop(all_gains, (-n_select_gain, gain, cmbs, deepcopy(nodes)))
                    else:
                        heapq.heappush(all_gains, (-i, gain, cmbs, deepcopy(nodes)))

            for _, gain, cmbs, nodes in all_gains:
                value, interaction = self.get_value_and_interaction(siv_scores, cmbs)
                feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(cmbs)])     
                children = [nodes[c] for c in cmbs]
                nodes[cmbs] = Node(
                    name=feature_name, 
                    score=value, 
                    interaction=interaction, 
                    children=children, 
                    k=k,
                    gain=gain
                )

                # add impossibles cmbs
                for c in cmbs:
                    nodes.pop(c)
                self.infos[k]['gain'].append(gain)
                self.infos[k]['nodes'].append(nodes)
                nodes_to_run = self.get_nodes_to_run(nodes=nodes, n_select_scores=n_select_scores)
                self.infos[k]['nodes_to_run'].append(nodes_to_run)

            if self.verbose:
                print(f'Scores: {[round(x, 4) for x in self.infos[k]["performance"]]}')
            return self._build(siv_scores, n_select_scores, n_select_gain, k+1)
    
    def get_nodes_to_run(self, nodes, n_select_scores):
        nodes_to_run = [(key, n.score) for key, n in nodes.items()] 
            
        if self.nodes_to_run_method == 'random':
            nodes_to_run = list(nodes.keys())
            filtered_nodes_to_run = self._random_selection(keys=nodes_to_run, n_select=n_select_scores)
        elif self.nodes_to_run_method == 'sort':
            nodes_to_run = [(key, n.score) for key, n in nodes.items()] 
            sorted_nodes_to_run = sorted(nodes_to_run, key=lambda x: x[1], reverse=True)
            filtered_nodes_to_run = list(map(lambda x: x[0], sorted_nodes_to_run[:n_select_scores]))
        elif self.nodes_to_run_method == 'full':
            filtered_nodes_to_run = list(nodes.keys())
        else:
            raise KeyError('`nodes_to_run_method` should be "random", "sort" or "full"')
        
        return filtered_nodes_to_run

    def filter_scores(self, scores, n_select_scores):
        if len(scores) == 1:
            return list(scores.keys())

        if self.filter_method == 'random':
            filtered = self._random_selection(keys=list(scores.keys()), n_select=n_select_scores)
        elif self.filter_method == 'sort':
            filtered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            filtered = list(map(lambda x: x[0], filtered[:n_select_scores]))
        elif self.filter_method == 'prob':
            if len(scores) < n_select_scores:
                filtered = list(scores.keys())
            else:
                keys, values = list(zip(*scores.items()))
                prob = np.array(values) / np.array(values).sum()
                idxes = np.random.choice(np.arange(len(prob)), size=(n_select_scores,), p=prob, replace=False)
                filtered = [keys[i] for i in idxes]
        else:
            raise KeyError('`filter_method` should be "random", "sort" or "prob')

        return filtered

    def _random_selection(self, keys: List[int], n_select: int):
        keys_to_run = deepcopy(keys)
        n = 0
        res = []
        while (n < n_select) and len(keys_to_run) > 0:
            c = random.choice(keys_to_run)
            keys_to_run.remove(c)
            res.append(c)
            n += 1
        return res

        
    def get_value_and_interaction(self, siv_scores, cmbs):
        r_l, c_l = np.tril_indices(siv_scores.shape[1], -1)
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