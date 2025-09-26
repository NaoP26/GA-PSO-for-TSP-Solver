import random
import numpy as np
import time
import matplotlib.pyplot as plt

# City names
city_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Fixed distance matrix (10 cities)
distance_matrix = np.array([
    [0, 29, 20, 21, 16, 31, 100, 45, 67, 72],
    [29, 0, 15, 17, 28, 23, 85, 55, 33, 66],
    [20, 15, 0, 30, 26, 40, 75, 43, 22, 50],
    [21, 17, 30, 0, 18, 35, 65, 48, 56, 41],
    [16, 28, 26, 18, 0, 25, 90, 62, 44, 53],
    [31, 23, 40, 35, 25, 0, 95, 57, 38, 64],
    [100, 85, 75, 65, 90, 95, 0, 78, 60, 80],
    [45, 55, 43, 48, 62, 57, 78, 0, 20, 34],
    [67, 33, 22, 56, 44, 38, 60, 20, 0, 28],
    [72, 66, 50, 41, 53, 64, 80, 34, 28, 0]
])

# Parameters
num_cities = len(distance_matrix)  # Number of cities
population_size = 100  # Population size
num_generations = 300  # Number of generations
mutation_rate = 0.1  # Mutation rate
elitism_count = 10  # Number of elites to carry over to the next generation

# Fitness calculation
def calculate_distance(path):
    """Calculate the total distance of a given path."""
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i], path[i + 1]]
    distance += distance_matrix[path[-1], path[0]]  # Return to the starting point
    return distance

# Create initial population
def create_population():
    """Generate a population of random paths."""
    return [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

# Select parents using roulette wheel selection
def select_parents(population, fitness):
    """Select two parents based on their fitness using roulette wheel selection."""
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    chosen_indices = np.random.choice(np.arange(len(population)), size=2, replace=False, p=probabilities)
    parent1 = population[chosen_indices[0]]
    parent2 = population[chosen_indices[1]]
    return parent1, parent2

# Crossover function
def crossover(parent1, parent2):
    """Perform ordered crossover to create a child."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    pointer = end
    for gene in parent2:
        if gene not in child:
            if pointer == size:
                pointer = 0
            child[pointer] = gene
            pointer += 1
    return child

# Mutation function
def mutate(individual):
    """Perform mutation by swapping two cities in the path."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# GA function with elitism
def genetic_algorithm():
    """Run the Genetic Algorithm for TSP with elitism."""
    population = create_population()
    best_distances = []
    best_paths = []

    start_time = time.time()

    for generation in range(num_generations):
        # Calculate fitness (inverse of the distance)
        fitness = [1 / calculate_distance(individual) for individual in population]
        new_population = []

        # Apply elitism: carry over the best 'elitism_count' individuals
        elites = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)[:elitism_count]
        new_population.extend([e[1] for e in elites])

        # Create the new generation
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population, fitness)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population  # Update population

        # Track the best solution
        best_index = np.argmax(fitness)
        best_distances.append(1 / fitness[best_index])
        best_paths.append(population[best_index])

    execution_time = time.time() - start_time

    # Convert the best path from indices to city names
    best_path = [city_names[i] for i in best_paths[np.argmin(best_distances)]]

    # Print results
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Best Path: {' -> '.join(best_path)}")
    print(f"Best Distance: {min(best_distances):.2f}")

    # Plot the convergence graph
    plot_results(best_distances)

def plot_results(distances):
    """Plot the convergence of the algorithm."""
    plt.figure(figsize=(8, 6))
    plt.plot(distances, label="Best Distance")
    plt.title("Convergence of Genetic Algorithm for TSP with Elitism")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend()
    plt.show()

genetic_algorithm()
