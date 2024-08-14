import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def calculate_cost(permutation, D, F):
    cost = 0
    for i in range(len(permutation)):
        for j in range(len(permutation)):
            cost += D[i][j] * F[permutation[i]][permutation[j]]
    return cost

def initialize_population(pop_size, n):
    population = []
    for _ in range(pop_size):
        individual = np.random.permutation(n)
        population.append(individual)
    return population

def evaluate_population(population, D, F):
    fitness_scores = []
    for individual in population:
        cost = calculate_cost(individual, D, F)
        fitness_scores.append(cost)
    return fitness_scores

def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

def crossover(parent1, parent2):
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    for item in parent2:
        if item not in child:
            for i in range(size):
                if child[i] == -1:
                    child[i] = item
                    break
    return child

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(D, F, pop_size=50, generations=100, mutation_rate=0.1, tournament_size=3):
    population = initialize_population(pop_size, len(D))
    best_solution = None
    best_cost = float('inf')
    costs = []

    for generation in range(generations):
        fitness_scores = evaluate_population(population, D, F)
        costs.append(min(fitness_scores))

        if min(fitness_scores) < best_cost:
            best_cost = min(fitness_scores)
            best_solution = population[fitness_scores.index(best_cost)]
        
        selected_population = tournament_selection(population, fitness_scores, tournament_size)
        
        new_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i+1]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population

    return best_solution, best_cost, costs

n = 5  # Number of places and facilities
pop_size = 50  # Number of permutations in population of one generation
generations = 25  # Number of generations
mutation_rate = 0.25  # Probability of mutation
tournament_size = 3   # Number of permutations (individuals) in one tournament

# Distance matrix
D = np.random.randint(1, 100, size=(n, n))
D = (D + D.T) // 2  
np.fill_diagonal(D, 0)


# Flow matrix
F = np.random.randint(1, 100, size=(n, n))

best_solution, best_cost, costs = genetic_algorithm(D, F, pop_size, generations, mutation_rate, tournament_size)

best_solution = [x + 1 for x in best_solution] # so the first index is 1 not 0

for i, facility in enumerate(best_solution):
    print(f"Na místo {i+1} je přiřazeno zařízení {facility}")

df_D = pd.DataFrame(D, columns=[f'Place {i+1}' for i in range(n)], index=[f'Place {i+1}' for i in range(n)])
df_F = pd.DataFrame(F, columns=[f'Facility {i+1}' for i in range(n)], index=[f'Facility {i+1}' for i in range(n)])

print("Vzdálenostní matice (D):")
print(df_D)
print("Toková matice (F):")
print(df_F)

print("Optimální řešení je:", best_solution)
print("Nejnižší náklady jsou:", best_cost)

for i, facility in enumerate(best_solution):
    print(f"Na místo {i+1} je přiřazeno zařízení {facility}.")
    

plt.figure(figsize=(12, 6))
plt.plot(range(len(costs)), costs, marker='o', linestyle='-', color='b')
plt.title('Náklady v průběhu generací')
plt.xlabel('Generace')
plt.ylabel('Náklady')
plt.grid(True)
plt.show()


