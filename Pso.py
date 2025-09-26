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
num_particles = 100  # Number of particles
num_generations = 300  # Number of generations
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient

# Fitness calculation
def calculate_distance(path):
    """Calculate the total distance of a given path."""
    distance = 0
    for i in range(len(path) - 1):
        distance += distance_matrix[path[i], path[i + 1]]
    distance += distance_matrix[path[-1], path[0]]  # Return to the starting point
    return distance

# Initialize particles
def initialize_particles():
    """Initialize particles with random paths."""
    particles = [np.array(random.sample(range(num_cities), num_cities)) for _ in range(num_particles)]
    velocities = [np.zeros(num_cities) for _ in range(num_particles)]
    return particles, velocities

# PSO update function
def update_particle(particle, velocity, best_position, global_best_position):
    """Update the particle's position based on velocity and best known positions."""
    inertia = w * velocity
    cognitive = c1 * random.random() * (best_position - particle)
    social = c2 * random.random() * (global_best_position - particle)

    new_velocity = inertia + cognitive + social
    new_particle = particle + new_velocity
    new_particle = np.clip(new_particle, 0, num_cities-1)  # Ensure positions are within bounds

    # Correct path to ensure no duplicate cities
    new_particle = correct_path(new_particle)
    return new_particle, new_velocity

# Correct path by ensuring no duplicate cities
def correct_path(path):
    """Ensure no city is visited more than once."""
    path = np.round(path).astype(int)  # Ensure integer indices
    unique_cities = np.unique(path)
    missing_cities = set(range(num_cities)) - set(unique_cities)
    missing_cities = list(missing_cities)

    for i, city in enumerate(path):
        if path.tolist().count(city) > 1:
            path[i] = missing_cities.pop(0)

    return path

# PSO function with Elitism
def particle_swarm_optimization():
    """Run PSO for TSP."""
    particles, velocities = initialize_particles()
    best_positions = particles.copy()
    global_best_position = particles[np.argmin([calculate_distance(p) for p in particles])]
    best_distances = []

    start_time = time.time()

    for generation in range(num_generations):
        # Evaluate fitness and update the best positions of particles
        for i, particle in enumerate(particles):
            fitness = calculate_distance(particle)
            if fitness < calculate_distance(best_positions[i]):
                best_positions[i] = particle

        # Update global best position (elitism - carry forward the best solution)
        current_best_particle = particles[np.argmin([calculate_distance(p) for p in particles])]
        current_best_distance = calculate_distance(current_best_particle)
        if current_best_distance < calculate_distance(global_best_position):
            global_best_position = current_best_particle

        # Update particle positions and velocities
        for i in range(num_particles):
            particles[i], velocities[i] = update_particle(particles[i], velocities[i], best_positions[i], global_best_position)

        # Track the best solution in this generation
        best_distances.append(calculate_distance(global_best_position))

    execution_time = time.time() - start_time

    # Convert the best path from indices to city names
    best_path = [city_names[i] for i in global_best_position]

    # Print results
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Best Path: {' -> '.join(best_path)}")
    print(f"Best Distance: {calculate_distance(global_best_position):.2f}")

    # Plot the convergence graph
    plot_results(best_distances)

def plot_results(distances):
    """Plot the convergence of the algorithm."""
    plt.figure(figsize=(8, 6))
    plt.plot(distances, label="Best Distance")
    plt.title("Convergence of PSO for TSP")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend()
    plt.show()

particle_swarm_optimization()