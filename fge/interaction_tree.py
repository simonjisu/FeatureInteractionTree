
import numpy as np
import pygraphviz
from io import BytesIO
from PIL import Image as PILImage
from anytree import Node, RenderTree, LevelGroupOrderIter

class ShapInteractionTree():
    def __init__(self, root):
        assert isinstance(root, Node), f'no a proper type of `root`: should be {Node}'
        self.root = root
        self.levels = None
        self.depth = None
        self.levels_reverse = None
        self._level_group_order()

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

    def _level_group_order(self):
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
            tree_str += f'{pre}{node.name}(v={node.score:.4f}, i={node.interaction:.4f})\n'
        return tree_str
    
    def get_performance_gap(self):
        gaps = [(node.k, node.gap) for *_, node in RenderTree(self.root) if node.k != 0]
        gaps = sorted(gaps, key=lambda x: x[0])
        return gaps

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
        s += f'<TR><TD>score={node.score:.4f}</TD></TR>'
        if node.interaction != 0.0:
            children_interaction = np.sum([child.interaction for child in node.children])
            s += f'<TR><TD>interaction={node.interaction - children_interaction:.4f}</TD></TR>'
        if node.gap is not None:
            s += f'<TR><TD>gap={node.gap:.6f}</TD></TR>'
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
            G.add_subgraph([node for node in G.nodes() if "+"not in node], rank="same")

        return G

    def _get_node_edge_key(self, node, parent=None):
        if node.score < 0.0:
            v_key = -1
        elif node.score > 0.0:
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