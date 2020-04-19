import pygad
import numpy

def genetic_algorithm(function, Lengths):
    
    fitness = 0
    count = 0
    
    Length = sum(Lengths)
    
    def fitness_func(solution):
        fitness = 1.0/function(solution)
        return fitness
    
    ga_instance = pygad.GA(num_generations=50, 
          sol_per_pop=8, 
          num_parents_mating=4, 
          num_genes=Length,
          fitness_func=fitness_func)
    
    while fitness < 1000 and count < 100:
        ga_instance.run()
        res = ga_instance.best_solution()
        x, fitness = res
        if count == 99: print('Max iterations reached') 
    
    return x