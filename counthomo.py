import itertools
import math
from decimal import Decimal, Context, InvalidOperation
import decimal
from typing import Union, Optional, Dict, Any

# Import SageMath types if available (requires SageMath installation)
try:
    from sage.graphs.graph import Graph as SageGraph
    from sage.graphs.digraph import DiGraph as SageDiGraph
except ImportError:
    # Define dummy types if SageMath is not installed, to allow code to parse
    class SageGraph:
        def vertices(self): return []
        def edges(self, labels=False): return []

    class SageDiGraph:
        def vertices(self): return []
        def edges(self, labels=False): return []


class WeightedGraph:
    def __init__(self, edge_weights: Dict[Any, Dict[Any, Union[float, Decimal]]], node_weights: Optional[Dict[Any, Union[float, Decimal]]] = None):
        """
        Represents a weighted graph using Decimal for precision.

        Args:
            edge_weights: A dictionary of dictionaries, where edge_weights[u][v] is the weight
                          of the directed edge from node u to node v. A weight of 0 implies no edge.
                          Can accept float or Decimal, stored as Decimal. Node keys can be of any hashable type.
            node_weights: Optional. A dictionary mapping node index (any hashable type) to its weight.
                          If None, all node weights default to Decimal(1.0). Can accept float or Decimal,
                          stored as Decimal.
        """
        # Convert all weights to Decimal and handle non-integer keys
        self.edge_weights = {}
        inferred_vertices = set()
        for u, v_weights in edge_weights.items():
            # Ensure u is hashable (SageMath nodes can be diverse)
            # For simplicity, assuming they are directly usable as dict keys
            self.edge_weights[u] = {}
            inferred_vertices.add(u)
            for v, weight in v_weights.items():
                 # Ensure v is hashable
                 # For simplicity, assuming they are directly usable as dict keys
                 self.edge_weights[u][v] = Decimal(str(weight))
                 inferred_vertices.add(v)

        if node_weights is None:
            self.node_weights = {v: Decimal(1.0) for v in inferred_vertices}
        else:
            self.node_weights = {v: Decimal(str(weight)) for v, weight in node_weights.items()}
            # Ensure all inferred vertices from edges are in node_weights, default to 1.0 if not provided
            for v in inferred_vertices:
                if v not in self.node_weights:
                    self.node_weights[v] = Decimal(1.0)

        # Ensure consistent vertex ordering from node_weights, which now includes all relevant vertices
        # Vertices can be any hashable type, so sorting might not be meaningful or possible.
        # Store them as a list in the order they were added or inferred, or just as a set.
        # Let's keep them as a list for consistent iteration order, but acknowledge it might not be sorted if types are mixed.
        self.vertices = list(self.node_weights.keys())
        # If sorting is desired and possible:
        # try:
        #     self.vertices = sorted(list(self.node_weights.keys()))
        # except TypeError:
        #     self.vertices = list(self.node_weights.keys()) # Fallback for unsortable types


    def get_node_weight(self, v: Any) -> Decimal:
        """Gets the weight of a node."""
        return self.node_weights.get(v, Decimal(0.0))

    def get_edge_weight(self, u: Any, v: Any) -> Decimal:
        """
        Gets the weight of the directed edge from u to v.
        Handles potential absence of u or v in edge_weights.
        """
        return self.edge_weights.get(u, {}).get(v, Decimal(0.0))

def convert_sage_graph_to_weighted_graph(sage_graph: Union[SageGraph, SageDiGraph]) -> WeightedGraph:
    """
    Converts a SageMath graph (Graph or DiGraph) to a WeightedGraph object.

    Args:
        sage_graph: The input SageMath graph.

    Returns:
        A WeightedGraph representation of the input SageMath graph.
    """
    node_weights = {}
    edge_weights = {}

    # Assuming default node weight of 1.0 for Sage graphs if not explicitly available/used
    # SageMath vertices can be any hashable type
    for v in sage_graph.vertices():
        node_weights[v] = Decimal(1.0)

    # SageMath edges are typically (u, v, label). Label is None for unweighted.
    for u, v, label in sage_graph.edges(labels=True):
        # Use the edge label as the weight. If label is None, use 1.0 (for unweighted edges)
        try:
            weight = Decimal(str(label)) if label is not None else Decimal(1.0)
        except InvalidOperation:
             # Handle cases where label cannot be converted to Decimal
             print(f"Warning: Could not convert edge label {label} to Decimal. Using 1.0 instead.")
             weight = Decimal(1.0)

        if u not in edge_weights:
            edge_weights[u] = {}
        edge_weights[u][v] = weight
        # If it's an undirected SageGraph, add the reverse edge as well with the same weight
        if isinstance(sage_graph, SageGraph) and not isinstance(sage_graph, SageDiGraph):
             if v not in edge_weights:
                 edge_weights[v] = {}
             edge_weights[v][u] = weight

    return WeightedGraph(node_weights=node_weights, edge_weights=edge_weights)

def count_homomorphisms(F: Union[WeightedGraph, SageGraph, SageDiGraph], G: Union[WeightedGraph, SageGraph, SageDiGraph], precision: Optional[int] = None) -> Decimal:
    """
    Counts the number of homomorphisms from weighted graph F to weighted graph G using Decimal.
    Can accept either WeightedGraph objects or SageMath Graph/DiGraph objects.

    Args:
        F: The source graph (WeightedGraph or SageMath Graph/DiGraph).
        G: The target graph (WeightedGraph or SageMath Graph/DiGraph).
        precision: Optional. The number of decimal places for calculations.
                   If None, the default Decimal context is used.

    Returns:
        The total count of homomorphisms as a Decimal.
    """

    # Convert SageMath graphs to WeightedGraph objects if necessary
    if isinstance(F, (SageGraph, SageDiGraph)):
        F_weighted = convert_sage_graph_to_weighted_graph(F)
    else:
        # Ensure F is a WeightedGraph if not a SageGraph
        if not isinstance(F, WeightedGraph):
             raise TypeError("Input graph F must be a WeightedGraph or a SageMath Graph/DiGraph.")
        F_weighted = F

    if isinstance(G, (SageGraph, SageDiGraph)):
        G_weighted = convert_sage_graph_to_weighted_graph(G)
    else:
        # Ensure G is a WeightedGraph if not a SageGraph
        if not isinstance(G, WeightedGraph):
             raise TypeError("Input graph G must be a WeightedGraph or a SageMath Graph/DiGraph.")
        G_weighted = G

    if precision is not None:
        ctx = Context(prec=precision)
        # Store original context to restore later
        original_context = decimal.getcontext()
        decimal.setcontext(ctx)

    V_F = F_weighted.vertices
    V_G = G_weighted.vertices
    total_hom = Decimal(0.0)

    # Generate all possible mappings from V(F) to V(G)
    # Use itertools.product to generate all possible mappings from the vertices of F to the vertices of G.
    # itertools.product(V_G, repeat=len(V_F)) creates an iterator that yields tuples.
    # Each tuple has length len(V_F), and its elements are chosen from V_G.
    # This effectively generates all functions from V_F to V_G.
    for mapping_tuple in itertools.product(V_G, repeat=len(V_F)):
        phi = {V_F[i]: mapping_tuple[i] for i in range(len(V_F))}

        # Calculate alpha_phi = prod_{v in V(F)} [alpha_{phi(v)}(G)]^alpha_v(F)
        alpha_phi = Decimal(1.0)
        for v in V_F:
            alpha_v_F = F_weighted.get_node_weight(v)
            alpha_phi_v_G = G_weighted.get_node_weight(phi[v])

            try:
                # Handle 0^0 = 1. Assumes positive node weights for G, and non-negative for F as per text.
                if alpha_phi_v_G == Decimal(0.0) and alpha_v_F == Decimal(0.0):
                     power_term = Decimal(1.0)
                elif alpha_phi_v_G >= Decimal(0.0): # Base must be non-negative for real exponent using Decimal.power
                     power_term = alpha_phi_v_G ** alpha_v_F
                else:
                     # This case shouldn't happen based on the problem description (positive node weights for G)
                     # but handle it just in case.
                     power_term = Decimal(0.0)
            except InvalidOperation:
                 # Handle cases like 0.0**-1 or negative base with non-integer exponent
                 power_term = Decimal(0.0) # Or handle error appropriately
                 # print(f"Warning: InvalidOperation in node weight power calculation for phi({v}).") # Avoid printing in library code


            alpha_phi *= power_term

        # Calculate hom_phi(F, G) = prod_{uv in E(F)} [beta_{phi(u)phi(v)}(G)]^beta_uv(F)
        hom_phi_F_G = Decimal(1.0)
        # Iterate through all possible edges in F based on F_weighted.vertices
        # Only consider edges that exist in F (non-zero weight)
        for u in V_F:
            for v in V_F:
                beta_uv_F = F_weighted.get_edge_weight(u, v)
                if beta_uv_F != Decimal(0.0):
                    target_u = phi[u]
                    target_v = phi[v]
                    beta_target_uv_G = G_weighted.get_edge_weight(target_u, target_v)

                    try:
                        # Handle 0^0 = 1. Assumes non-negative edge weights for G as per text.
                        if beta_target_uv_G == Decimal(0.0) and beta_uv_F == Decimal(0.0):
                            power_term = Decimal(1.0)
                        elif beta_target_uv_G >= Decimal(0.0): # Base must be non-negative for real exponent using Decimal.power
                             power_term = beta_target_uv_G ** beta_uv_F
                        else:
                            # This case shouldn't happen based on problem description (non-negative edge weights for G)
                            # but handle it just in case.
                            power_term = Decimal(0.0)
                    except InvalidOperation:
                        # Handle cases like 0.0**-1 or negative base with non-integer exponent
                        power_term = Decimal(0.0) # Or handle error appropriately
                        # print(f"Warning: InvalidOperation in edge weight power calculation for edge ({target_u}, {target_v}) in G.") # Avoid printing in library code

                    hom_phi_F_G *= power_term


        total_hom += alpha_phi * hom_phi_F_G

    # Restore original context if it was set
    if precision is not None:
        decimal.setcontext(original_context)

    return total_hom
# Use the '**' operator for exponentiation, as the '.power()' method was reported as unavailable.
                     # This calculates the power of the target node weight raised to the source node weight, maintaining Decimal precision.