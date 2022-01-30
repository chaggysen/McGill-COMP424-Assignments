from math import sqrt
from itertools import permutations
import math
import numpy as np
import matplotlib.pyplot as plt
import random


def generate_tsp_instance(n):
    """
    Generate a TSP of n cities
    """
    cities = [None for _ in range(n)]
    random_x = np.random.random(size=n) 
    random_y = np.random.random(size=n)
    for i in range(n):
        cities[i] = (random_x[i], random_y[i])
    return cities


def euclidean_distance(city1, city2):
    """
    Returns the Euclidean distance between two coordinates
    """
    x1, x2, y1, y2 = city1[0], city2[0], city1[1], city2[1]
    delta_x, delta_y = x1 - x2, y1 - y2
    return sqrt((delta_x ** 2) + (delta_y ** 2))


def tsp_solver_brute_force(cities):
    """
    A brute force TSP solver using the first city as source
    """
    # data validation
    if cities is None or len(cities) == 0:
        return 0
    MIN_LEN, optimal_path= float("inf"), None

    # Generate permutations
    next_permutation = permutations(cities)
    for perm in list(next_permutation):
        current_distance = get_distance(perm)
        if current_distance < MIN_LEN:
            MIN_LEN, optimal_path = current_distance, perm
    return [MIN_LEN, optimal_path]


def random_path(cities):
    """
    Generate a random tsp path for input cities
    """
    path = []
    while cities:
        random_city = cities[random.randint(0, len(cities) - 1)]
        path.append(random_city)
        cities.remove(random_city)
    return path


def get_distance(path):
    """
    Get the total distance of a path
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += euclidean_distance(path[i], path[i + 1])
    return distance + euclidean_distance(path[-1], path[0])

def tsp_random_solver(cities):
    path = random_path(cities)
    distance = get_distance(path)
    return [distance, path]


def get_neighbors(current_path):
    """
    Get all neighbours of a current path. 
    """
    neighbors = []
    for i in range(len(current_path)- 1):
        for k in range(i + 1, len(current_path)):
            neighbor = []
            # take route[0] to route[i-1] and add them in order to new_route
            for idx in range(i):
                neighbor.append(current_path[idx])
            # take route[i] to route[k] and add them in reverse order to new_route
            for idx in range(k, i - 1, -1):
                neighbor.append(current_path[idx])
            # take route[k+1] to end and add them in order to new_route
            for idx in range(k + 1, len(current_path)):
                neighbor.append(current_path[idx])
            neighbors.append(neighbor)
    return neighbors


def get_best_neighbor(neighbors):
    min_distance = get_distance(neighbors[0])
    best_neighbor = neighbors[0]
    for neighbor in neighbors:
        distance = get_distance(neighbor)
        if distance < min_distance:
            best_neighbor, min_distance = neighbor, distance
    return best_neighbor


def tsp_hill_climbing(starting_path, starting_distance):
    """
    Hill climbing algorithm
    """
    current_path, current_distance = starting_path, starting_distance

    neighbors = get_neighbors(current_path)
    best_neighbor = get_best_neighbor(neighbors)
    best_neighbor_distance = get_distance(best_neighbor)

    while best_neighbor_distance < current_distance:
        # update current path and distance
        current_path = best_neighbor
        current_distance = best_neighbor_distance

        neighbors = get_neighbors(current_path)
        best_neighbor = get_best_neighbor(neighbors)
        best_neighbor_distance = get_distance(best_neighbor)
    return current_distance, current_path


def plot_cities_and_path(path):
    """
    Plot the cities and the path
    """
    source = path[0]
    x, y = zip(*path)
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.plot(source[0], source[1], '.r', 20)
    plt.title("Path for TSP")
    plt.show()


def calculate_stats(distances):
    stats = {}
    stats["mean"] = np.mean(distances)
    stats["min"] = np.min(distances)
    stats["max"] = np.max(distances)
    stats["std"] = np.std(distances)
    return stats

def run_tsp(nb_cities_per_tsp):
    optimal_distances, random_distances, ai_distances = [], [], []
    optimal_ai_output_count, optimal_random_output_count = 0, 0

    for _ in range(100):
        print("======================================================================================================================")
        cities = generate_tsp_instance(nb_cities_per_tsp)

        # print generated tsp 
        print("========== ORIGINAL CITIES ==========")
        print(cities)

        if nb_cities_per_tsp < 10:
            # print the optimal solution using brute force
            print("========== OPTIMAL SOLUTION ==========")
            optimal_d, optimal_path = tsp_solver_brute_force(cities.copy())
            print(f"optimal distance: {optimal_d}")
            print(f"optimal path: {optimal_path}")
            optimal_distances.append(optimal_d)

        # print a random solution
        print("========== RANDOM SOLUTION ==========")
        random_d, random_path = tsp_random_solver(cities.copy())
        print(f"random distance: {random_d}")
        print(f"random path: {random_path}")
        random_distances.append(random_d)

        # print the ai solution
        print("========== AI SOLUTIONS ==========")
        ai_d, ai_path = tsp_hill_climbing(random_path, random_d)
        print(f"ai distance: {ai_d}")
        print(f"ai path: {ai_path}")
        ai_distances.append(ai_d)

        if nb_cities_per_tsp < 10 and math.isclose(optimal_d, ai_d, abs_tol=0.01):
            optimal_ai_output_count += 1
        if nb_cities_per_tsp < 10 and math.isclose(optimal_d, random_d, abs_tol=0.01):
            optimal_random_output_count += 1
    
    print("======================================== FINAL STATS =========================================")
    if nb_cities_per_tsp < 10:
        print("========== OPTIMAL SOLUTIONS STATS ==========")
        print(calculate_stats(optimal_distances))
    print("========== RANDOM SOLUTIONS STATS ==========")
    print(calculate_stats(random_distances))
    print("========== AI SOLUTIONS STATS ==========")
    print(calculate_stats(ai_distances))
    if nb_cities_per_tsp < 10:
        print("========== RANDOM FOUND OPTIMAL SOLUTION COUNT ==========")
        print(optimal_random_output_count)
        print("========== AI FOUND OPTIMAL SOLUTION COUNT ==========")
        print(optimal_ai_output_count)




def main():
    # will run brute force solution if number of city < 10
    run_tsp(7)

if __name__ == "__main__":
    main()
