
import numpy as np
import pandas as pd
import itertools

from .functions import *
from .utils import flatten

from typing import Dict, Tuple, Any, List, Set, Callable

from anytree import Node, RenderTree, LevelGroupOrderIter
# from anytree.exporter import DotExporter

# import networkx as nx
# from pyvis import network as net
from pathlib import Path

from copy import deepcopy
import pygraphviz
from io import BytesIO
from PIL import Image as PILImage

class TreeBuilder():
    def __init__(self):
        self.score_methods = {
            'base': g_base,
            'abs': g_abs,
            'abs_interaction': g_abs_only_interaction,
            'ratio': g_ratio,
        }
        self.cache_path = Path('./cache')
        if not self.cache_path.exists():
            self.cache_path.mkdir()
        
        self.reset_tree()
    
    def reset_tree(self):
        self.root = None
        self.levels = None
        self.depth = None
        self.levels_reverse = None

        # for visualization
        self._step = -1
        self._iterations = {}

    def level_group_order(self):
        assert self.root is not None, 'not root'
        self.levels = {}
        for i, childrens in enumerate(LevelGroupOrderIter(self.root)):
            self.levels[i] = []
            for node in childrens:
                self.levels[i].append(node.name)
        self.depth = len(self.levels)
        self.levels_reverse = {self.depth-k-1:v for k, v in self.levels.items()}

    def __repr__(self):
        if self.root is None:
            tree_str = str(self.__class__)
        else:
            tree_str = ''
            for pre, fill, node in RenderTree(self.root):
                tree_str += f'{pre}{node.name}(v={node.value:.4f}, s={node.score:.4f})\n'
        return tree_str

    def _apply_record(self, nodes):
        self._step += 1
        self._iterations[self._step] = deepcopy(nodes)

    def build(
        self, 
        method: str, 
        shap_interactions: np.ndarray, 
        feature_names: List[str] | pd.Index | None =None,
        magnitude: bool=False,
        top_n: int =1,
    ) -> List[Tuple[Any, Node]]:
        """Build a bottom-up tree

        Args:
            method (str): A method to process shap interaction values\n
                \t- `base`: shap interaction values\n
                \t- `abs`: absolute shap interaction values\n
                \t- `abs_interaction`: absolute shape interaction values without main effects\n
                \t- `ratio`: ratio of absolute shap interaction values to main effects\n
            shap_interactions (np.ndarray): Shap interaction values\n
            feature_names (List[str] | pd.Index | None, optional): Feature names. \n
                If given None, it will automatically generate feature name with numbers. Defaults to None.\n
            magnitude(bool): Calculate node's shap value with absolute values, without considering direction. 
                Defaults to False.\n
            top_n (int, optional): Top n scores to select from scores. Defaults to 1.\n

        """
        self.reset_tree()
        
        # feature settings
        if feature_names is None:
            self.show_features = False
            self.feature_names = np.arange(shap_interactions.shape[-1])
        else:
            self.show_features = True
            self.feature_names = feature_names
        g_fn = self.score_methods[method]
        nodes = self._build(shap_interactions, g_fn, top_n, magnitude)
        self.root = list(nodes.values())[-1]
        self.level_group_order()
        return list(nodes.items())

    def _build(self, shap_interactions: np.ndarray, g_fn: Callable, top_n: int, magnitude: bool) -> None:
        """Build a bottom-up tree

        Args:
            shap_interactions (np.ndarray): shap interaction values\n
            g_fn (Callable): A method function to process scores\n
            top_n (int): Top n scores to select from scores\n
            magnitude(bool): Calculate node's shap value with absolute values, without considering direction. 
                Defaults to False.\n

        """        
        if shap_interactions.ndim == 3:
            # ndim == 3 case: global tree
            build_global = True
            self.siv = shap_interactions.mean(0) if not magnitude else np.abs(shap_interactions).mean(0)
        elif shap_interactions.ndim == 2:
            # ndim == 2 case: single tree
            build_global = False
            self.siv = shap_interactions  if not magnitude else np.abs(shap_interactions)
        else:
            raise ValueError('number of dimension of `shap_interactions` should be 2 or 3')
        
        self.siv_scores = g_fn(shap_interactions, build_global)
        r_diag, c_diag = np.diag_indices(len(self.feature_names))
        main_effect = self.siv[r_diag, c_diag]
        main_scores = self.siv_scores[r_diag, c_diag]
        
        nodes = {}
        scores = {}
        values = {}
        done = set() # check need to run it or pass at the next time


        for i, name in enumerate(self.feature_names):
            nodes[i] = Node(name=name, parent=None, value=main_effect[i], score=main_scores[i])

        # visualization record
        self._apply_record(nodes)

        # build tree
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
        """Recursive function to build a bottom-up tree

        Args:
            nodes (Dict[Any, Node]): nodes with combinations of features as key, Node as value. \n
            scores (Dict[Any, float]): shap interaction scores with combinations of features as key, score as value. \n
            values (Dict[Any, float]): shap interaction values with combinations of features as key, summation of shap value as value. \n
            done (Set): record for used nodes. \n
            g_fn (Callable): a method function to process scores. \n
            top_n (int): top n scores to select from scores. \n

        """        
        nodes_to_run = [k for k in nodes.keys() if k not in done]
        if len(nodes_to_run) == 1:
            return nodes
        else:
            # update `scores` and `values`
            scores, values = self._get_scores(
                nodes_to_run, 
                siv_scores=self.siv_scores, 
                siv=self.siv, 
                scores=scores, 
                values=values
            )
            # select top_n nodes
            selected_cmbs = self._select(scores, top_n=top_n)
            for best_cmbs in selected_cmbs:
                # create node
                if self.feature_names is not None:
                    feature_name = '+'.join([str(self.feature_names[i]) for i in flatten(best_cmbs)])
                else:
                    feature_name = str(best_cmbs)
                
                children = [nodes[c] for c in best_cmbs]  
                nodes[best_cmbs] = Node(name=feature_name, value=values[best_cmbs], score=scores[best_cmbs], children=children)
                
                # add done and need to remove all impossible options for 'scores'
                for c in best_cmbs:
                    done.add(c)
                    impossible_coor = list(filter(lambda x: c in x, scores.keys()))
                    for coor in impossible_coor:
                        scores.pop(coor, None)
                        values.pop(coor, None)

                self._apply_record(nodes)
    
            return self._build_tree(nodes, scores, values, done, g_fn, top_n)

    def _get_scores(
        self,
        nodes_to_run: List[Tuple[Any] | int], 
        siv_scores: np.ndarray,
        siv: np.ndarray,
        scores: Dict[Tuple[Any], float],
        values: Dict[Tuple[Any], float],
    ):
        """Calcuate scores by each combination of trial nodes

        Args:
            nodes_to_run (List[Tuple[Any]  |  int]): Trial nodes to build a parent node. \n
            siv_scores (np.ndarray): shap interaction scores. \n
            siv (np.ndarray): shap interaction values. \n
            scores (Dict[Tuple[Any], float]): shap interaction scores with combinations of features as key, score as value. \n
            values (Dict[Tuple[Any], float]): shap interaction values with combinations of features as key, summation of shap value as value. \n

        """    
        for cmbs in itertools.combinations(nodes_to_run, 2):
            if cmbs not in scores.keys():
                r, c = list(zip(*itertools.product(flatten(cmbs), flatten(cmbs))))
                scores[cmbs] = siv_scores[r, c].sum()
                values[cmbs] = siv[r, c].sum()
        return scores, values

    def _select(self, scores: Dict[Any, float], top_n: int):
        """Select combinations of Top N scores 

        Args:
            scores (Dict[Any, float]): shap interaction scores with combinations of features as key, score as value. \n
            top_n (int): top n scores to select from scores. \n

        """
        l = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        i = 0
        selected = []
        while (i < top_n) and len(l) > 0:
            top_cmb, _ = l.pop(0)
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
        
    def show_tree(self, nodes):
        G = self._draw_graph(nodes)
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

    def _node_fmt(self, node):
        return f'{"+".join(node.name.split("/"))}\n value={node.value:.3f}\n score={node.score:.3f}'

    def _draw_graph(self, nodes):
        kwargs = {
            'node': {
                'fontsize': 12,
                'color': 'blue',
                'shape': 'box'
            },
            'edge': {
                'arrowsize': 0.5, 
                'headclip': True, 
                'tailclip': True
            }
        }

        if isinstance(nodes, list):
            nodes = dict(nodes)
        G = pygraphviz.AGraph(directed=False)
        G.graph_attr['rankdir'] = 'TB'
        G.graph_attr["ordering"] = "out"
        G.layout(prog='dot')

        for _, node in nodes.items():
            if node.parent is None:
                G.add_node(self._node_fmt(node), **kwargs['node'])
            else:
                G.add_node(self._node_fmt(node), **kwargs['node'])
                G.add_node(self._node_fmt(node.parent), **kwargs['node'])
                G.add_edge(self._node_fmt(node.parent), self._node_fmt(node), **kwargs['edge'])
        G.add_subgraph([node for node in G.nodes() if "+"not in node], rank="same")

        return G

    # def show(self, notebook=False, **kwargs):
    #     nt = self.export_pyvis(notebook=notebook)
    #     cur_path = Path('.').absolute()
    #     nt.show(str(cur_path / 'cache' / 'tree.html'))

    # def export_pyvis(self, notebook=True, **kwargs):
    #     if self.root is None:
    #         raise KeyError('There is no Node in the tree, try to call `TreeBuilder.build(method)` first')
        
    #     filename = 'tree.dot'
    #     self._export(typ='dot', filename=filename)

    #     nx_graph = nx.drawing.nx_agraph.read_dot(self.cache_path / filename)
    #     nt = net.Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=notebook)
    #     nt.from_nx(nx_graph)
    #     return nt

    # def _export(self, typ: str, filename: str):
    #     if self.root is None:
    #         raise KeyError('There is no Node in the tree, try to call `TreeBuilder.build(method)` first')
    #     exporter = DotExporter(
    #         self.root, 
    #         edgeattrfunc = lambda node, child: "dir=back"
    #     )
    #     if typ == 'dot':
    #         exporter.to_dotfile(self.cache_path / filename)
    #     elif typ == 'png':
    #         exporter.to_picture(self.cache_path / filename)
    #     else:
    #         raise KeyError('Wrong `typ`')