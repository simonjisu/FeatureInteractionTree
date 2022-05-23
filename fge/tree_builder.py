
import numpy as np
import itertools
from .functions import *
from typing import Dict, Tuple, Any, List, Set, Callable
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import pandas as pd
import networkx as nx
from pyvis import network as net
from pathlib import Path

def flatten(li: List):
    for ele in li:
        if isinstance(ele, list) or isinstance(ele, tuple):
            yield from flatten(ele)
        else:
            yield ele

class TreeBuilder():
    def __init__(self, shap_interactions: np.ndarray, feature_names: List[str] | pd.Index | None):
        if feature_names is None:
            self.show_features = False
            self.feature_names = np.arange(shap_interactions.shape[-1])
        else:
            self.show_features = True
            self.feature_names = feature_names
        if shap_interactions.ndim == 3:
            # given mulitple instance shap interactions, 
            # we will calculate by its' average impact value(dont care the direction)
            self.siv = np.abs(shap_interactions).mean(0)
        else:
            self.siv = shap_interactions
        self.g_functions = {
            'sum': g_sum
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
                tree_str += f'{pre}{node.name}({node.value:.4f})\n'
        return tree_str

    def build(self, method: str) -> List[Tuple[Any, Node]]:
        g_fn = self.g_functions[method]
        num_feature = len(self.feature_names)
        r_diag, c_diag = np.diag_indices(num_feature)
        main_effect = self.siv[r_diag, c_diag]

        scores = {}
        nodes = {}
        done = set()  # check need to run it or pass at the next time
        
        # initialize leaf-nodes
        for i, name in enumerate(self.feature_names):
            nodes[i] = Node(name=name, parent=None, value=main_effect[i])
        nodes = self._build_tree(nodes, scores, done, g_fn)
        self.root = list(nodes.values())[-1]
        return list(nodes.items())

    def _build_tree(self, nodes: Dict[Any, Node], scores: Dict[Any, float], done: Set, g_fn: Callable):
        nodes_to_run = [k for k in nodes.keys() if k not in done]
        if len(nodes_to_run) == 1:
            return nodes
        else: 
            for cmbs in itertools.combinations(nodes_to_run, 2):
                if cmbs not in scores.keys():
                    r, c = list(zip(*itertools.product(flatten(cmbs), flatten(cmbs))))
                    scores[cmbs] = self.siv[r, c].sum()

            best_cmbs, max_value = g_fn(scores)
            # get feature names
            if len(list(flatten(best_cmbs))) != len(self.feature_names):
                if self.show_features:
                    feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(best_cmbs)])
                else:
                    feature_name = f'f{len(list(flatten(best_cmbs)))-1}'
            else:
                feature_name = 'all'
            children = []
            for c in best_cmbs:
                children.append(nodes[c])
                done.add(c)
                # need to remove all impossible options for 'scores'
                impossible_coor = list(filter(lambda x: c in x, scores.keys()))
                for coor in impossible_coor:
                    scores.pop(coor, None)

            nodes[best_cmbs] = Node(name=feature_name, value=max_value, children=children)
            return self._build_tree(nodes, scores, done, g_fn)

    def show(self, notebook: bool=False, **kwargs):
        nt = self.export_pyviz(notebook=notebook)
        cur_path = Path('.').absolute()
        nt.show(str(cur_path / 'cache' / 'tree.html'))

    def export_pyviz(self, notebook: bool=False, **kwargs):
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