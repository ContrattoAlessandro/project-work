import numpy as np
from Problem import Problem
from src.solver import Solver

def solution(p: Problem):
    solver = Solver(p, pop_size=1000, generations=100)
    best_cost, trips = solver.solve()
    
    # Extract gold information from the graph for fast lookup
    g = p.graph
    n_nodes = g.number_of_nodes()
    gold_amounts = np.array([g.nodes[i].get('gold', 0.0) for i in range(n_nodes)])
    
    path = [(0, 0)]
    
    for segment, k in trips:
        # segment is the list of cities visited in this trip
        # k is the number of times this trip is repeated
        
        if k < 1: continue 
        
        # If we make k identical trips, in each one we collect 1/k of the total gold
        gold_fraction = 1.0 / k
        
        single_trip = []
        for city in segment:
            amount_collected = gold_amounts[city] * gold_fraction
            single_trip.append((int(city), float(amount_collected)))
        
        # Each trip ends by returning to the Hub
        single_trip.append((0, 0))
        
        # Add this trip to the main path k times
        for _ in range(k):
            path.extend(single_trip)

    return path
