
from mimetypes import init
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

from typing import Dict, Any
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
            n_select_scores: int=5,
            n_select_performance: int=5, 
            max_iter: int | None=None,
            initialize: str | None='random',  # random / sort / full
            filter_method: str = 'random',  # random / sort
            rt_only_best: bool=True,
            verbose: bool=False
        ):
        """
        filter_method(sort) + score_method(only interaction method: abs_interaction, ratio)
        """

        self.reset_tree(verbose)
        # feature settings
        self.feature_names = np.arange(shap_interactions.shape[-1])
        self.initialize = initialize
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
        # self.infos[k]['done'] = [set()]
        for i, name in enumerate(self.feature_names):
            self.infos[k]['nodes'][0][i] = Node(
                name=str(name), parent=None, score=main_effect[i], interaction=0.0, k=0, drop=0.0
            )

        # TODO: Select Random Nodes to start? or Start from largest Interactions? or Main Effects?
        # try to filter start nodes with larger main effect, since they have larger impact
        if initialize == 'random':
            filtered_nodes_to_run = list(np.random.choice(self.feature_names, size=(n_select_scores,), replace=False))
        elif initialize == 'sort':
            nodes_to_run = [(key, n.score) for key, n in self.infos[k]['nodes'][0].items()] # [(key, n.score) for key, n in self.infos[k]['nodes'][0].items() if key not in self.infos[k]['done'][0]]
            sorted_nodes_to_run = sorted(nodes_to_run, key=lambda x: x[1], reverse=True)
            filtered_nodes_to_run = list(map(lambda x: x[0], sorted_nodes_to_run[:n_select_scores]))
        elif initialize == 'full':
            filtered_nodes_to_run = list(self.feature_names)
        else:
            raise KeyError('`initialize` should be "random", "sort" or "full"')

        self.infos[k]['nodes_to_run'] = [filtered_nodes_to_run]
        self.infos[k]['performance'] = self.polyfitter.original_score
        self._build(siv_scores, n_select_scores, n_select_performance, k+1)
        if not self.verbose:
            self.logger.close()
        if rt_only_best:
            return self._get_best_tree()
        else:
            return self._get_all_trees()

    def _build(self, siv_scores, n_select_scores, n_select_performance, k):
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
                self.infos[k]['performance'] = []
                
            performances = []
            heapq.heapify(performances)
            i = 0
            while prev_nodes_to_run:
                nodes_to_run = prev_nodes_to_run.pop(0)
                nodes = prev_nodes.pop(0)
                
                scores = self.get_scores(siv_scores, nodes_to_run)
                filtered_keys = self.filter_scores(scores, n_select_scores)
                if self.verbose:
                    print(f'Number of filtered keys: {len(filtered_keys)}')
                else:
                    self.logger.set_postfix_str(f'N keys to run: {len(filtered_keys)}')
                
                for cmbs in filtered_keys:
                    i += 1
                    trials = list(nodes.keys()) + [cmbs]
                    performance = self.polyfitter.fit_selected(trials)
                    performance_drop = self.infos[0]['performance'] - performance
                    if len(performances) >= n_select_performance:
                        if performances[0][1] > performance_drop:
                            heapq.heappushpop(performances, (-n_select_performance, performance_drop, cmbs, deepcopy(nodes)))
                    else:
                        heapq.heappush(performances, (-i, performance_drop, cmbs, deepcopy(nodes)))

            for _, p, cmbs, nodes in performances:
                value, interaction = self.get_value_and_interaction(siv_scores, cmbs)
                feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(cmbs)])     
                children = [nodes[c] for c in cmbs]
                nodes[cmbs] = Node(
                    name=feature_name, 
                    score=value, 
                    interaction=interaction, 
                    children=children, 
                    k=k,
                    drop=0.0
                )
                nodes[cmbs].drop = p

                # add impossibles cmbs
                for c in cmbs:
                    nodes.pop(c)
                self.infos[k]['performance'].append(p)
                self.infos[k]['nodes'].append(nodes)
                self.infos[k]['nodes_to_run'].append(list(nodes.keys()))

            if self.verbose:
                print(f'Scores: {[round(x, 4) for x in self.infos[k]["performance"]]}')
            return self._build(siv_scores, n_select_scores, n_select_performance, k+1)
    
    def filter_scores(self, scores, n_select_scores):
        if len(scores) == 1:
            return list(scores.keys())

        if self.filter_method == 'random':
            scores_to_run = deepcopy(list(scores.keys()))
            # dup = set()
            n = 0
            filtered = []
            while (n < n_select_scores) and scores_to_run: #(len(dup) < len(scores)):
                c = random.choice(scores_to_run)
                scores_to_run.remove(c)
                filtered.append(c)
                # if c in dup:
                #     continue
                # else:
                #     dup.add(c)
                #     filtered.append(c)
                n += 1
        elif self.filter_method == 'sort':
            filtered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            filtered = list(map(lambda x: x[0], filtered[:n_select_scores]))
        else:
            raise KeyError('`filter_method` should be "random" or "sort"')

        return filtered

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