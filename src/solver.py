import networkx as nx
import numpy as np
import random
import multiprocessing as mp
from numba import njit
from tqdm import tqdm
from scipy.sparse import csgraph

# global variables for worker processes
worker_dist_matrix = None
worker_gold = None
worker_alpha = None
worker_beta = None
worker_window_size = None

def init_worker(dist_matrix, gold, alpha, beta, window_size):
    """Initialize worker process with shared data"""
    global worker_dist_matrix, worker_gold, worker_alpha, worker_beta, worker_window_size
    worker_dist_matrix = dist_matrix
    worker_gold = gold
    worker_alpha = alpha
    worker_beta = beta
    worker_window_size = window_size

def worker_batch(parents_batch):
    """Worker function to create and evaluate a batch of offspring"""
    results = []
    for p1, p2 in parents_batch:
        # Crossover and Mutation
        child = crossover_scx(p1, p2, worker_dist_matrix)
        mutate_insert(child)
        
        # Evaluate
        cost, P, S = split(child, worker_dist_matrix, worker_gold,
                           worker_alpha, worker_beta, worker_window_size)
        results.append((cost, child, P, S))
    return results

@njit(cache=True)
def optimal_k(dist, gold, alpha, beta):
    """
    Calculates the optimal number of trips (k) to transport gold from a single city to the hub, 
    minimizing the total cost given the load penalty
    """
    exponent = 1.0 / beta
    k_star = alpha * gold * (dist ** (1.0 - exponent)) * (((beta - 1.0) / 2.0) ** exponent)
    
    # k must be a positive integer
    k = int(k_star)
    if k < 1: 
        k = 1
    
    # Compare the actual costs of the two integers closest to k* (k and k+1) to find the discrete global minimum.
    
    # Cost for k trips
    c1 = k * (2.0 * dist + (alpha * dist * (gold / k)) ** beta)
    
    # Cost for k+1 trips
    k2 = k + 1
    c2 = k2 * (2.0 * dist + (alpha * dist * (gold / k2)) ** beta)
    
    if c1 <= c2:
        return k, c1
    return k2, c2

@njit(cache=True)
def split(individual, dist_matrix, gold, alpha, beta, window_size):
    """
    Prins' Algorithm
    """
    n = len(individual)
    V = np.full(n + 1, np.inf)
    P = np.zeros(n + 1, dtype=np.int32)
    S = np.ones(n + 1, dtype=np.int32)
    V[0] = 0.0
    
    for i in range(n):
        if V[i] == np.inf:
            continue

        load = 0.0
        cost_accum = 0.0
        u = 0

        max_j = min(n, i + window_size)

        for j in range(i, max_j):
            v = individual[j]
            d_uv = dist_matrix[u, v]
            # Cost of edge u -> v
            cost_accum += d_uv + (alpha * d_uv * load) ** beta
            # Item collection
            load += gold[v]
            # Hypothetical return cost v -> Hub
            d_vh = dist_matrix[v, 0]
            cost_return = d_vh + (alpha * d_vh * load) ** beta
            total_trip_cost = cost_accum + cost_return
            
            # Optimal split
            best_k = 1
            if beta > 1.0 and i == j: # Single city trip
                k, k_cost = optimal_k(d_vh, gold[v], alpha, beta)
                if k > 1 and k_cost < total_trip_cost:
                    total_trip_cost = k_cost
                    best_k = k
            
            # Found a cheaper way to reach city j+1
            if V[i] + total_trip_cost < V[j+1]:
                V[j+1] = V[i] + total_trip_cost
                P[j+1] = i
                S[j+1] = best_k
            u = v
    return V[n], P, S

@njit(cache=True)
def crossover_scx(p1, p2, dist_matrix):
    """
    Constructs an offspring by choosing the shortest edge among the candidates proposed by the two parents
    """
    size = len(p1)
    offspring = np.empty(size, dtype=np.int32)
    
    # Identify node range for arrays
    max_node = max(p1.max(), p2.max())
    visited = np.zeros(max_node + 1, dtype=np.bool_)
    
    # Inverse maps allow instant lookup of a city's position in the parent
    p1_indices = np.full(max_node + 1, -1, dtype=np.int32)
    p2_indices = np.full(max_node + 1, -1, dtype=np.int32)
    for i in range(size):
        p1_indices[p1[i]] = i
        p2_indices[p2[i]] = i
        
    # The offspring always starts with the first city of the first parent
    current_node = p1[0]
    offspring[0] = current_node
    visited[current_node] = True
    
    # Progressive construction of the offspring
    for count in range(1, size):
        idx1 = p1_indices[current_node]
        idx2 = p2_indices[current_node]
        
        # Identify candidates (the next city in the parents' chromosomes)
        next1 = p1[0] if idx1 == size - 1 else p1[idx1 + 1]
        next2 = p2[0] if idx2 == size - 1 else p2[idx2 + 1]
        
        # Evaluate costs of proposed edges, if already visited cost is infinite
        cost1 = dist_matrix[current_node, next1] if not visited[next1] else np.inf
        cost2 = dist_matrix[current_node, next2] if not visited[next2] else np.inf
            
        if cost1 == np.inf and cost2 == np.inf:
            # If both candidates have already been visited, search for the first free city
            for i in range(size):
                if not visited[p1[i]]:
                    next_node = p1[i]
                    break
        elif cost1 <= cost2:
            next_node = next1
        else:
            next_node = next2
                
        # Add the chosen city to the offspring and proceed to the next step
        offspring[count] = next_node
        visited[next_node] = True
        current_node = next_node
        
    return offspring

@njit(cache=True)
def mutate_insert(individual):
    """
    Extracts a random city and inserts it at a random position
    """
    size = len(individual)
    if size < 2:
        return
        
    # Choose a random position and extract the city
    remove_pos = random.randint(0, size - 1)
    element = individual[remove_pos]

    # Shift all next cities to the left
    for i in range(remove_pos, size - 1):
        individual[i] = individual[i+1]

    # Choose a new random position for insertion
    insert_pos = random.randint(0, size - 1)
    
    # Shift cities to the right from the end up to the new position to make space
    for i in range(size - 2, insert_pos - 1, -1):
            individual[i+1] = individual[i]
            
    # Insert the extracted city into the new position
    individual[insert_pos] = element

class Solver:
    def __init__(self, problem, pop_size=100, generations=300):
        self.problem = problem
        self.graph = problem.graph
        self.pop_size = pop_size
        self.generations = generations
        self.alpha = problem.alpha
        self.beta = problem.beta
        self.window_size = 50 if self.beta < 1.0 else 20
        
        # Extract problem data into Numpy arrays
        n_nodes = self.graph.number_of_nodes()
        self.cities = np.arange(1, n_nodes, dtype=np.int32)
        self.gold = np.array([self.graph.nodes[i].get('gold', 0) for i in range(n_nodes)])
        
        # Pre-calculation of the distance matrix using Dijkstra
        adj_matrix = nx.to_scipy_sparse_array(self.graph, weight='dist')
        self.dist_matrix = csgraph.shortest_path(adj_matrix, directed=False)

    def evaluate(self, ind):
        return split(ind, self.dist_matrix, self.gold, self.alpha, self.beta, self.window_size)

    def reconstruct_solution(self, individual, P, S):
        trips = []
        tour = list(individual)
        # Backtrack from the end of the tour to the beginning
        current_pos = len(tour)
        while current_pos > 0:
            previous_pos = P[current_pos]
            k = S[current_pos]
            # Extract the segment and its optimal number of splits (k)
            segment = tour[previous_pos : current_pos]
            trips.append((segment, k))
            # Move back to the start of this segment
            current_pos = previous_pos
        # Reverse the list since we backtracked from last to first
        trips.reverse()
        return trips

    def solve(self):
        # Create initial population randomly
        pop = []
        for _ in range(self.pop_size):
            ind = np.random.permutation(self.cities)
            cost, P, S = self.evaluate(ind)
            pop.append((cost, ind, P, S))
        pop.sort(key=lambda x: x[0])
        
        best = pop[0]

        n_offspring = self.pop_size * 7
        n_workers = mp.cpu_count()
        
        # Create pool with initialized workers
        with mp.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(self.dist_matrix, self.gold, self.alpha, self.beta, self.window_size)
        ) as pool:
            
            batch_size = max(1, n_offspring // n_workers)
            
            pbar = tqdm(range(self.generations), desc="Generations")
            for gen in pbar:
                pbar.set_postfix({"Best Cost": f"{best[0]:,.0f}"})
                
                pop_inds = [x[1] for x in pop]
                pop_fits = np.array([x[0] for x in pop])
                
                # Create parent pairs
                parent_pairs = []
                for _ in range(n_offspring):
                    idx1 = np.random.randint(0, self.pop_size, 5)
                    idx2 = np.random.randint(0, self.pop_size, 5)
                    p1 = pop_inds[idx1[np.argmin(pop_fits[idx1])]]
                    p2 = pop_inds[idx2[np.argmin(pop_fits[idx2])]]
                    parent_pairs.append((p1, p2))
                
                batches = [parent_pairs[i:i + batch_size] for i in range(0, len(parent_pairs), batch_size)]
                
                # Evaluate offspring
                results_nested = pool.map(worker_batch, batches)
                offspring = [item for sublist in results_nested for item in sublist]
                
                # Select best
                pop = sorted(pop + offspring, key=lambda x: x[0])[:self.pop_size]
                
                if pop[0][0] < best[0]:
                    best = pop[0]
        
        # Return cost and reconstruction of the best path found using stored P and S
        return best[0], self.reconstruct_solution(best[1], best[2], best[3])
