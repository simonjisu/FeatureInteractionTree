
import numpy as np
import pandas as pd
import itertools

from .functions import *
from .utils import flatten
from .fitter import PolyFitter

from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import pygraphviz
from io import BytesIO
from PIL import Image as PILImage

from anytree import Node, RenderTree, LevelGroupOrderIter
from copy import deepcopy

class TreeBuilder():
    def __init__(self, dataset: Dict[str, Any], task_type: str, original_score: None|float=None):
        
        self.score_methods = {
            'base': g_base,
            'abs': g_abs,
            'abs_interaction': g_abs_only_interaction,
            'ratio': g_ratio,
        }
        self.polyfitter = PolyFitter(dataset, task_type, original_score)

    def reset_tree(self, verbose):
        self.root = None
        self.infos = defaultdict(dict)
        self.logger = tqdm(desc='Building Tree')
        self.verbose = verbose

    def build(
            self, 
            score_method: str, 
            shap_interactions: np.ndarray, 
            n_select: int=5, 
            max_iter: int | None=None,
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
            self.siv = shap_interactions.mean(0)
        elif shap_interactions.ndim == 2:
            # ndim == 2 case: single tree
            build_global = False
            self.siv = shap_interactions
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
        main_effect = self.siv[r_diag, c_diag]
        self.infos[k]['nodes'] = [dict()]
        self.infos[k]['done'] = [set()]
        for i, name in enumerate(self.feature_names):
            self.infos[k]['nodes'][0][i] = Node(name=name, parent=None, value=main_effect[i], interaction=0.0, k=0)

        nodes_to_run = [key for key in self.infos[k]['nodes'][0].keys() if key not in self.infos[k]['done'][0]]
        self.infos[k]['nodes_to_run'] = [nodes_to_run]
        self.infos[k]['performance'] = self.polyfitter.original_score
        self._build(siv_scores, n_select, k+1)
        self.logger.close()
        trees = [(tuple(x.items())[-1][0], ShapInteractionTree(root=tuple(x.items())[-1][1])) 
            for x in list(self.infos.items())[-1][1]['nodes']]
        return trees
        
    def _build(self, siv_scores, n_select, k):
        self.logger.update(1)
        prev_nodes_to_run = deepcopy(self.infos[k-1]['nodes_to_run'])
        prev_nodes = deepcopy(self.infos[k-1]['nodes'])
        prev_dones = deepcopy(self.infos[k-1]['done'])
        
        if (not prev_nodes_to_run) or (k == self.max_iter):
            return 
        else:
            if self.verbose:
                print(f'Current Step: {k}')
                for n in prev_nodes_to_run:
                    print(f'Nodes to Run: {n}')
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
                scores = self.get_scores(nodes_to_run, siv_scores)
                filtered_keys = self.filter_scores(scores, existed_cmbs)
                for cmbs in filtered_keys:
                    new_nodes = deepcopy(nodes)
                    new_done = deepcopy(done)

                    value, interaction = self.get_value_and_interaction(cmbs)
                    
                    feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(cmbs)])
                    
                    children = [new_nodes[c] for c in cmbs]
                    new_nodes[cmbs] = Node(name=feature_name, value=value, interaction=interaction, children=children, k=k)
                    self.infos[k]['nodes'].append(new_nodes)

                    # add impossibles
                    for c in cmbs:
                        new_done.add(c)

                    self.infos[k]['done'].append(new_done)
                    new_nodes_to_run = [k for k in new_nodes.keys() if k not in new_done]
                    self.infos[k]['nodes_to_run'].append(new_nodes_to_run)

            # eval
            for nodes in self.infos[k]['nodes']:
                performance = self.polyfitter.fit_selected(nodes, self.feature_names)
                self.infos[k]['performance'].append(self.infos[0]['performance'] - performance)

            sorted_idx = np.argsort(self.infos[k]['performance'])#[::-1]

            self.infos[k]['nodes_to_run'] = np.array(self.infos[k]['nodes_to_run'], dtype=object)[sorted_idx[:n_select]].tolist()
            self.infos[k]['nodes'] = np.array(self.infos[k]['nodes'], dtype=object)[sorted_idx[:n_select]].tolist()
            self.infos[k]['done'] = np.array(self.infos[k]['done'], dtype=object)[sorted_idx[:n_select]].tolist()
            self.infos[k]['performance'] = np.array(self.infos[k]['performance'])[sorted_idx[:n_select]].tolist()
            
            if self.verbose:
                print(f'Scores: {[round(x, 4) for x in self.infos[k]["performance"]]}')
            return self._build(siv_scores, n_select, k+1)
    
    def filter_scores(self, scores, existed_cmbs=None):
        if len(scores) == 1:
            return list(scores.keys())
        filtered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if existed_cmbs is not None:
            filtered = [(key, value) for key, value in filtered if key not in existed_cmbs]
        return list(map(lambda x: x[0], filtered[:int(len(filtered)/2)]))

    def get_value_and_interaction(self, cmbs):
        r_l, c_l = np.tril_indices(self.n_features, -1)
        cmbs_flattend = list(flatten(cmbs))
        cmbs_idx = np.arange(len(r_l))[np.isin(r_l, cmbs_flattend) & np.isin(c_l, cmbs_flattend)]

        r, c = list(zip(*itertools.product(flatten(cmbs), flatten(cmbs))))
        value = self.siv[r, c].sum()
        interaction = self.siv[r_l, c_l][cmbs_idx].sum()
        return value, interaction

    def get_scores(self, nodes_to_run, siv_scores):
        scores = {}
        for cs in itertools.combinations(nodes_to_run, 2):
            if cs not in scores.keys():
                r, c = list(zip(*itertools.product(flatten(cs), flatten(cs))))
                scores[cs] = siv_scores[r, c].sum()
        return scores

class ShapInteractionTree():
    def __init__(self, root):
        assert isinstance(root, Node), f'no a proper type of `root`: should be {Node}'
        self.root = root
        self.levels = None
        self.depth = None
        self.levels_reverse = None
        self.level_group_order()

        self.colors = {
            'blue': '#2F33BD', 'red': '#C9352C', 'black': '#000000'
        }
        self.show_kwargs = {
            'node':{
                1: {'fontname': 'Arial', 'fontsize': 12, 'color': self.colors['red'], 'shape': 'box'},
                0: {'fontname': 'Arial', 'fontsize': 12, 'color': self.colors['black'], 'shape': 'box'},
                -1: {'fontname': 'Arial', 'fontsize': 12, 'color': self.colors['blue'], 'shape': 'box'}
            },
            'edge': {
                1: {'color': self.colors['red'], 'arrowsize': 0.5, 'headclip': True, 'tailclip': True},
                0: {'color': self.colors['black'], 'arrowsize': 0.5, 'headclip': True, 'tailclip': True},
                -1: {'color': self.colors['blue'], 'arrowsize': 0.5, 'headclip': True, 'tailclip': True}
            }
        }

    def level_group_order(self):
        self.levels = {}
        for i, childrens in enumerate(LevelGroupOrderIter(self.root)):
            self.levels[i] = []
            for node in childrens:
                self.levels[i].append(node.name)
        self.depth = len(self.levels)
        self.levels_reverse = {self.depth-k-1:v for k, v in self.levels.items()}

    def __repr__(self):
        tree_str = ''
        for pre, fill, node in RenderTree(self.root):
            tree_str += f'{pre}{node.name}(v={node.value:.4f}, i={node.interaction:.4f})\n'
        return tree_str
  
    def show_tree(self, feature_names=None): 
        G = self._draw_graph(feature_names)
        img = self._display_graph(G)
        return img

    def show_step_by_step(self, step: int):
        return self.show_tree(self._iterations[step])

    def _display_graph(self, G):
        # https://github.com/chebee7i/nxpd/blob/master/nxpd/ipythonsupport.py
        imgbuf = BytesIO()
        G.draw(imgbuf, format='png', prog='dot')
        img = PILImage.open(imgbuf)
        return img

    def _fmt(self, node, feature_names=None):
        s = '< <TABLE BORDER="0" ALIGN="CENTER">'
        if feature_names is not None:
            fs = [feature_names[int(n)] for n in node.name.split("+")]
        else:
            fs = node.name.split("/")
        fs_str = '' if node.k == 0 else f'({node.k}) '
        fs_str += f'{fs[0]} + ... + {fs[-1]}' if len(fs) > 2 else "+".join(fs)
        s += f'<TR><TD><B>{fs_str}</B></TD></TR>'
        s += f'<TR><TD>value={node.value:.4f}</TD></TR>'
        if node.interaction != 0.0:
            s += f'<TR><TD>interaction={node.interaction:.4f}</TD></TR>'
        return s + '</TABLE> >'

    def _draw_graph(self, feature_names=None, i=None):
        # http://www.graphviz.org/doc/info/attrs.html#k:style
        G = pygraphviz.AGraph(directed=True)
        G.graph_attr['rankdir'] = 'BT'
        G.graph_attr["ordering"] = "out"
        G.layout(prog='dot')

        for *_, node in RenderTree(self.root):
            if node.parent is None:
                # root
                v_key, _ = self._get_node_edge_key(node, parent=None)
                G.add_node(node, label=self._fmt(node, feature_names), **self.show_kwargs['node'][v_key])
            else:
                v_key, e_key = self._get_node_edge_key(node, parent=node.parent)
                G.add_node(node, label=self._fmt(node, feature_names), **self.show_kwargs['node'][v_key])
                G.add_edge(node, node.parent, **self.show_kwargs['edge'][e_key])
        # G.add_subgraph([node for node in G.nodes() if "+"not in node], rank="same")

        return G

    def _get_node_edge_key(self, node, parent=None):
        if node.value < 0.0:
            v_key = -1
        elif node.value > 0.0:
            v_key = 1
        else:
            v_key = 0
        
        if parent is not None:    
            if parent.interaction < 0.0:
                e_key = -1
            elif parent.interaction > 0.0:
                e_key = 1
            else:
                e_key = 0
        else:
            e_key = None
        return v_key, e_key