from math import sqrt
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import random


def generate_tsp_instance(n):
    """
    Generate a TSP of n cities
    """
    cities = [None for _ in range(n)]
    random_x = np.random.randint(1000, size=n)
    random_y = np.random.randint(1000, size=n)
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


def tsp_solver_brute_force(cities, source=0):
    """
    A brute force TSP solver using the first city as source
    """
    # data validation
    if cities is None or len(cities) == 0:
        return 0

    # build destination list
    destinations = []
    for i in range(len(cities)):
        if i != source:
            destinations.append(cities[i])

    source_city = cities[source]
    MIN_LEN = float("inf")
    optimal_path = None

    # Generate permutations
    next_permutation = permutations(destinations)
    current_city = source_city
    for perm in list(next_permutation):
        current_distance = 0
        for city in perm:
            current_distance += euclidean_distance(city, current_city)
            current_city = city
        current_distance += euclidean_distance(current_city, source_city)
        if current_distance < MIN_LEN:
            MIN_LEN = current_distance
            optimal_path = perm
    path_list = list(optimal_path)
    path_list.insert(0, source_city)
    path_list.append(source_city)
    return [MIN_LEN, path_list]


def random_path(cities):
    """
    Generate a random tsp path for input cities
    """
    path = []
    origin = cities[0]
    path.append(origin)
    cities.remove(origin)
    while cities:
        random_city = cities[random.randint(0, len(cities) - 1)]
        path.append(random_city)
        cities.remove(random_city)
    path.append(origin)
    return path


def get_distance(path):
    """
    Get the total distance of a path
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += euclidean_distance(path[i], path[i + 1])
    return distance


def tsp_hill_climbing(cities):
    """
    Hill climbing algorithm
    """
    current_path = random_path(cities)
    current_distance = get_distance(cities)

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


def main():
    optimal_distances = []
    ai_distances = []
    optimal_ai_output_count = 0
    for _ in range(100):
        cities = generate_tsp_instance(7)
        print("========== ORIGINAL CITIES ==========")
        print(cities)
        optimal_d, optimal_path = tsp_solver_brute_force(cities)
        print("========== OPTIMAL SOLUTIONS ==========")
        print(optimal_d)
        print(optimal_path)
        optimal_distances.append(optimal_d)
        ai_d, ai_path = tsp_hill_climbing(cities)
        print("========== AI SOLUTIONS ==========")
        print(ai_d)
        print(ai_path)
        if optimal_d == ai_d:
            optimal_ai_output_count += 1
    print("========== OPTIMAL SOLUTIONS STATS ==========")
    print("========== AI SOLUTIONS STATS ==========")
    print(calculate_stats(optimal_distances))
    # plot_cities_and_path(path)


if __name__ == "__main__":
    main()
