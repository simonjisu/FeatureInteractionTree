
import numpy as np
import pandas as pd

from .functions import *
from .utils import flatten
from typing import Dict, Tuple, Any, List, Set, Callable
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

import networkx as nx
from pyvis import network as net
from pathlib import Path

class TreeBuilder():
    def __init__(self):

        self.g_functions = {
            'base': g_base
        }
        self.cache_path = Path('./cache')
        if not self.cache_path.exists():
            self.cache_path.mkdir()
        self.root = None

    def __repr__(self):
        if self.root is None:
            tree_str = str(self.__class__)
        else:
            tree_str = ''
            for pre, fill, node in RenderTree(self.root):
                tree_str += f'{pre}{node.name}(v={node.value:.2f}, s={node.score:.2f})\n'
        return tree_str

    def build(
        self, 
        method: str, 
        shap_interactions: np.ndarray, 
        feature_names: List[str] | pd.Index | None =None,
        magnitude: bool =True,
        top_n: int =1,
    ) -> List[Tuple[Any, Node]]:
        # feature settings
        if feature_names is None:
            self.show_features = False
            self.feature_names = np.arange(shap_interactions.shape[-1])
        else:
            self.show_features = True
            self.feature_names = feature_names

        if shap_interactions.ndim == 3:
            # ndim == 3 case: global tree
            build_global = True
        elif shap_interactions.ndim == 2:
            # ndim == 2 case: single tree
            build_global = False
        else:
            raise ValueError('number of dimension of `shap_interactions` should be 2 or 3')
        g_fn = self.g_functions[method]
        nodes = self._build(shap_interactions, g_fn, magnitude, build_global, top_n)
        self.root = list(nodes.values())[-1]
        return list(nodes.items())

    def _build(self, shap_interactions: np.ndarray, g_fn: Callable, magnitude: bool, build_global: bool, top_n: int) -> None:
        if build_global:
            self.siv_scores = np.abs(shap_interactions).mean(0) if magnitude else shap_interactions.mean(0)
            # TODO: is it right to show the mean absolute values?
            # because each instance tree will be built differently
            # 1. build a single tree using abs(interactions)
            # 2. build multiple trees and combine them together(how to measure the tree structure similarity?)
            # currently doing 1st method
            self.siv = np.abs(shap_interactions).mean(0)  
        else:
            self.siv_scores = np.abs(shap_interactions) if magnitude else shap_interactions
            self.siv = shap_interactions

        r_diag, c_diag = np.diag_indices(len(self.feature_names))
        main_effect = self.siv[r_diag, c_diag]
        main_scores = self.siv_scores[r_diag, c_diag]
        
        nodes = {}
        scores = {}
        values = {}
        done = set() # check need to run it or pass at the next time

        for i, name in enumerate(self.feature_names):
            nodes[i] = Node(name=name, parent=None, value=main_effect[i], score=main_scores[i])
        nodes = self._build_tree(nodes, scores, values, done, g_fn, top_n)
        return nodes

    def _build_tree(
            self, 
            nodes: Dict[Any, Node], 
            scores: Dict[Any, float], 
            values: Dict[Any, float], 
            done: Set, 
            g_fn: Callable, 
            top_n: int
        ):
        nodes_to_run = [k for k in nodes.keys() if k not in done]
        if len(nodes_to_run) == 1:
            return nodes
        else:
            # apply g_fn -> update `scores` and `values`
            scores, values = g_fn(nodes_to_run, siv_scores=self.siv_scores, siv=self.siv, scores=scores, values=values)
            # can select multiple nodes
            selected_cmbs = self._select(scores, top_n=top_n)
            for best_cmbs in selected_cmbs:
                # create node
                if self.feature_names is not None:
                    feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(best_cmbs)])
                else:
                    feature_name = str(best_cmbs)
                
                children = [nodes[c] for c in best_cmbs]  
                nodes[best_cmbs] = Node(name=feature_name, value=values[best_cmbs], score=scores[best_cmbs], children=children)

                for c in best_cmbs:
                    done.add(c)
                    # need to remove all impossible options for 'scores'
                    impossible_coor = list(filter(lambda x: c in x, scores.keys()))
                    for coor in impossible_coor:
                        scores.pop(coor, None)
                        values.pop(coor, None)

            return self._build_tree(nodes, scores, values, done, g_fn, top_n)

    def _select(self, scores: Dict[Any, float], top_n: int):
        l = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        i = 0
        selected = []
        while (i < top_n) and len(l) > 0:
            top_cmb, top_score = l.pop(0)
            selected.append(top_cmb)
            # filter out impossible combinations
            all_impossibles = set()
            for c in top_cmb:
                impossible_coor = list(filter(lambda x: c in x, scores.keys()))
                for coor in impossible_coor:
                    all_impossibles.add(coor)
            l = list(filter(lambda x: x[0] not in all_impossibles, l))
            i += 1
        return selected

    def show(self, notebook=False, **kwargs):
        nt = self.export_pyvis(notebook=notebook)
        cur_path = Path('.').absolute()
        nt.show(str(cur_path / 'cache' / 'tree.html'))

    def export_pyvis(self, notebook=True, **kwargs):
        if self.root is None:
            raise KeyError('There is no Node in the tree, try to call `TreeBuilder.build(method)` first')
        
        filename = 'tree.dot'
        self._export(typ='dot', filename=filename)

        nx_graph = nx.drawing.nx_agraph.read_dot(self.cache_path / filename)
        nt = net.Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=notebook)
        nt.from_nx(nx_graph)
        return nt

    def _export(self, typ: str, filename: str):
        if self.root is None:
            raise KeyError('There is no Node in the tree, try to call `TreeBuilder.build(method)` first')
        exporter = DotExporter(
            self.root, 
            edgeattrfunc = lambda node, child: "dir=back"
        )
        if typ == 'dot':
            exporter.to_dotfile(self.cache_path / filename)
        elif typ == 'png':
            exporter.to_picture(self.cache_path / filename)
        else:
            raise KeyError('Wrong `typ`')   