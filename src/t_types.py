#import typing as typ
from networkx import DiGraph
from collections import defaultdict

type t_node = int
type t_weight = float
type t_graph = dict[t_node, dict[t_node, t_weight]]  # [u][v][weight]
type t_DynGraph = dict[int, DiGraph]

type t_query = int
type t_time_query  = dict[int, t_query]
type t_query_dict = dict[int, t_time_query]

type t_edge = tuple[t_node, t_node]

type t_changes_dict = dict[str, set[t_node] | set[t_edge]]

type t_walks_dict = dict[t_node, list[list[int]]]
type t_edge_to_walk_dict = defaultdict[tuple[int, int], list[list[int]]]