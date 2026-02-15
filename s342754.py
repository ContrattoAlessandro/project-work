import numpy as np
import networkx as nx
from Problem import Problem
from src.solver import Solver

def solution(p: Problem):
    solver = Solver(p, pop_size=1000, generations=100)
    best_cost, trips = solver.solve()
    
    # Extract gold information from the graph for fast lookup
    g = p.graph
    n_nodes = g.number_of_nodes()
    gold_amounts = np.array([g.nodes[i].get('gold', 0.0) for i in range(n_nodes)])
    
    # Precompute shortest paths
    shortest_paths = dict(nx.all_pairs_dijkstra_path(g, weight='dist'))
    
    path = [(0, 0)]
    
    for segment, k in trips:
        # segment is the list of cities visited in this trip
        # k is the number of times this trip is repeated
        
        if k < 1: continue 
        
        # If we make k identical trips, in each one we collect 1/k of the total gold
        gold_fraction = 1.0 / k
        
        # We need to expand each step into the actual shortest path in the graph.
        full_route = [0] + list(segment) + [0]
        
        single_trip = []
        for step_idx in range(len(full_route) - 1):
            src = full_route[step_idx]
            dst = full_route[step_idx + 1]
            
            # Get the actual shortest path between src and dst
            actual_path = shortest_paths[src][dst]
            
            # Add each intermediate node (skip the first as it's already in the path)
            for node in actual_path[1:]:
                # Only collect gold at the actual destination cities in the segment
                if node == dst and node != 0:
                    amount_collected = gold_amounts[node] * gold_fraction
                else:
                    amount_collected = 0.0
                single_trip.append((int(node), float(amount_collected)))
        
        # Add this trip to the main path k times
        for _ in range(k):
            path.extend(single_trip)
    
    return path
