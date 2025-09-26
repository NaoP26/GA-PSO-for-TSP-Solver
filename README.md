# 🧬 GA & PSO Metaheuristic Solvers for the Traveling Salesperson Problem (TSP)

This repository contains **Python** implementations of two popular metaheuristic optimization techniques—the **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)**—applied to solve the classic **Traveling Salesperson Problem (TSP)**.

The project is designed to demonstrate how these population-based, nature-inspired algorithms can be used to find near-optimal solutions for NP-hard combinatorial optimization challenges.

-----

## ✨ Project Files and Algorithms

The core of the project consists of two distinct solver scripts. Both scripts are configured to optimize the path for a predefined 10-city distance matrix.

| File | Algorithm | Description |
| :--- | :--- | :--- |
| **`Ga.py`** | **Genetic Algorithm** | Solves the TSP using principles of evolutionary computing, including selection, crossover, and mutation to generate better routes across generations. |
| **`Pso.py`** | **Particle Swarm Optimization** | Solves the TSP by modeling the movement of a swarm of particles (potential solutions), guided by their own best-found positions (`pBest`) and the global best position (`gBest`). |

-----

## 🚀 Getting Started

### Prerequisites

You need **Python 3.x** and the following scientific computing and plotting libraries installed to run the scripts:

```bash
pip install numpy matplotlib
```

### How to Run

Execute the optimization algorithms directly from your terminal:

1.  **Run the Genetic Algorithm:**

    ```bash
    python Ga.py
    ```

2.  **Run the Particle Swarm Optimization:**

    ```bash
    python Pso.py
    ```

Each script will automatically initialize the problem, run the optimization simulation for a fixed number of generations/iterations, and display the results.

-----

## 📈 Expected Results

Upon execution, the scripts will output the final optimized route and distance to the console, and generate a plot showing the algorithm's convergence (how the best distance decreased over time).

**Example Console Output:**

```
Execution time: 0.1234 seconds
Best Path: A -> C -> D -> E -> F -> J -> I -> G -> H -> B -> A
Best Distance: 256.78
```

A **Matplotlib window** will display the convergence plot, illustrating the optimization process.

-----

## ⚙️ Customization

The core parameters for both algorithms are defined at the beginning of their respective Python files. You can easily modify them to test different scenarios and tune performance:

| Algorithm | Key Parameters to Adjust |
| :--- | :--- |
| **GA.py** | `population_size`, `generations`, `mutation_rate`, `elitism_count` |
| **PSO.py** | `num_particles`, `max_iterations`, `w` (inertia), `c1`, `c2` (acceleration factors) |

-----

## 📧 Contact

  * **Author:** Ömer Opan
