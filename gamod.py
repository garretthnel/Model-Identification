import pygad
import numpy
from tqdm.auto import tqdm

def genetic_algorithm(function, Orders):
    
    fitness = 0
    count = 0
    
    Length = sum(Orders)
    
    def fitness_func(solution, solution_idx):
        fitness = 1.0/function(solution)
        return fitness
    
    ga_instance = pygad.GA(num_generations=50, 
          sol_per_pop=8, 
          num_parents_mating=4, 
          num_genes=Length,
          fitness_func=fitness_func)
    
    f = 100
    with tqdm(total=f) as pbar:
        while fitness < 1/1e-2 and count < f:
            ga_instance.run()
            res = ga_instance.best_solution()
            x, fitness, solution_idx = res
            count += 1
            pbar.update(1)
    
    return x