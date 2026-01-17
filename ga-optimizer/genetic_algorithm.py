import numpy as np
import random


def create_individual():
    short = random.randint(5, 30)
    long = random.randint(short + 10, 100)
    return (short, long)


def create_population(size):
    return [create_individual() for _ in range(size)]


def crossover(parent1, parent2):
    child = (parent1[0], parent2[1])
    if child[0] >= child[1]:
        child = (child[0], child[0] + 15)
    return child


def mutate(individual, mutation_rate=0.2):
    if random.random() < mutation_rate:
        short = individual[0] + random.randint(-5, 5)
        short = max(3, min(50, short))
        long = individual[1] + random.randint(-10, 10)
        long = max(short + 5, min(150, long))
        return (short, long)
    return individual


def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(list(zip(population, fitness_scores)), 3)
        winner = max(tournament, key=lambda x: x[1])
        parents.append(winner[0])
    return parents


def run_ga(fitness_func, pop_size=20, generations=15):
    population = create_population(pop_size)
    
    # Track all individuals per generation for visualization
    all_generations = []
    
    for gen in range(generations):
        fitness_scores = [fitness_func(ind) for ind in population]
        best_fitness = max(fitness_scores)
        best_ind = population[fitness_scores.index(best_fitness)]
        
        # Store this generation's data
        gen_data = {
            'population': population.copy(),
            'fitness': fitness_scores.copy(),
            'best': best_fitness
        }
        all_generations.append(gen_data)
        
        print(f"Gen {gen+1}: Best ROI = {best_fitness:.2f}% | Params = {best_ind}")
        
        parents = select_parents(population, fitness_scores, pop_size // 2)
        
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        
        population = new_pop
    
    final_scores = [fitness_func(ind) for ind in population]
    best_idx = final_scores.index(max(final_scores))
    
    return population[best_idx], all_generations
