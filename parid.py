import scipy
import psopy
import gamod
import tsmod

import time
import os.path

import random
import pandas
import numpy
import control

from matplotlib import pyplot as plt

class Estimation:
    def __init__(self):
        self.input = {'Step': self.step, 
                      'Rect': self.rect, 
                      'Doublet': self.doublet}
        
        self.noise = {'None': self.no_noise,
                      'Uniform': self.uniform_noise,
                      'Non-uniform': self.pseudo_noise}
        
        self.tech = {'Differential Evolution': self.DE,
                     'Least Squares': self.LS,
                     'Scipy Minimize': self.MIN,
                     'Particle Swarm': self.PSO,
                     'Genetic Algorithm': self.GA,
                     'Tabu Search': self.TS}
        
        self.description = {'input': '',
                            'noise': '',
                            'tech': ''}
        
        self.reset_history()
        self.set_id()
        
        
        if os.path.isfile('data.csv'):
            self.data = pandas.read_csv('data.csv').set_index('key')
        else:
            self.data = pandas.DataFrame({'key': ['Dur', 'Num', 'Den', 'Num Est', 'Den Est', 'Time', 'Y', 'U', 'Y Est', 'Hist']}).set_index('key')
            self.data.to_csv('data.csv')
        
        self.num = [] 
        self.den = [] 
        self.numest = []
        self.denest = []
        
        return None
    
    def set_id(self, save=''):
        if save != '':
            self.current_id = f'{self.description['tech']}_{self.description['input']}_{self.description['noise']}_{save}'
        else:
            self.current_id = f'{self.description['tech']}_{self.description['input']}_{self.description['noise']}'
    
    def load_data(self, file):
        if file == '':
            print('Please enter a file name to load.')
            return [0, 0, 0]
        else:
            df = pandas.read_csv(f'{file}.csv')
            self.y = df['Y'] - df['Y'][0]
            self.u = df['U'] - df['U'][0]
            self.t = df['Time']
            
            plt.plot(self.t, self.y, label="Real/Loaded")
            plt.legend()
            plt.show()
            
            return self.y, self.u, self.t
    
    def save_data(self, save_name=''):
#         ['Duration', 'Num', 'Den', 'Num Est', 'Den Est', 'Time', 'Y', 'U', 'Y Est', 'History']
        
        self.set_id(save_name)
        
        for i in range(len(self.coeff)):
            if i < self.div:
                self.numest.append(self.coeff[i])
            else:
                self.denest.append(self.coeff[i])
    
        d = {f'{self.current_id}': [self.dur, self.num, self.den, self.numest, self.denest, self.t, self.y, self.u, self.yr, self.history]}

        df = pandas.DataFrame(d)
        
        self.data = pandas.concat([self.data, df], axis=1)
        
        self.data.to_csv('data.csv')

        return print(f'{self.current_id} data successfully saved.')
    
    def step(self, t, start, stop):
        self.description['input'] = 'S'
        
        if t<start:
            return 0
        else:
            return 1

    def rect(self, t, start, drop, stop):
        self.description['input'] = 'R'
        
        if t>=start and t<stop:
            return 1
        else:
            return 0

    def doublet(self, t, start, drop, rise, stop):
        self.description['input'] = 'D'
        
        if t>=start and t<drop:
            return 1
        elif t>=drop and t<rise:
            return -1
        else:
            return 0
    
    def uniform_noise(self, ydata, Magnitude):
        self.description['noise'] = 'U'
        return [ydata[i] + random.uniform(-1,1)*Magnitude for i in range(len(ydata))]

    def pseudo_noise(self, ydata, Magnitude):
        self.description['noise'] = 'P'
        return [ydata[i] + random.randrange(start=-1,stop=1)*Magnitude for i in range(len(ydata))]

    def no_noise(self, ydata):
        self.description['noise'] = 'N'
        return ydata
    
    def create_system(self, Numerator, Denominator):
        self.num = Numerator
        self.den = Denominator
        return control.tf(self.num, self.den)
        
    def DE(self, function, Bounds):
        self.coeff = scipy.optimize.differential_evolution(function, Bounds).x
        self.description['tech'] = 'DE'
        return self.coeff

    def LS(self, function, Initial_Guess):
        self.coeff = scipy.optimize.least_squares(function, [i[0] for i in Initial_Guess]).x
        self.description['tech'] = 'LS'
        return self.coeff

    def MIN(self, function, Initial_Guess):
        self.coeff = scipy.optimize.minimize(function, [i[0] for i in Initial_Guess]).x
        self.description['tech'] = 'MIN'
        return self.coeff

    def PSO(self, function, Initial_Bounds):
        self.coeff = psopy.minimize(function, numpy.array([numpy.random.uniform(*b, 10) for b in Initial_Bounds]).T).x
        self.description['tech'] = 'PSO'
        return self.coeff

    def GA(self, function, Lengths):
        self.coeff = gamod.genetic_algorithm(function, [int(i[0]) for i in Lengths])
        self.description['tech'] = 'GA'
        return self.coeff

    def TS(self, function, Bounds):
        self.coeff = tsmod.tabu_search(function, Bounds)
        self.description['tech'] = 'TS'
        return self.coeff
        
    def err(self, params, div):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=self.t, U=self.u)
        self.history.append(sum((self.y - ysim)**2))
        return self.history[-1]

    def res(self, params, div):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=self.t, U=self.u)
        plt.plot(tsim, ysim, '--', label="Estimation")
        plt.legend()            
        plt.show()
        return ysim
    
    def response(self, system, t, u):
        return control.forced_response(system, T=t, U=u)
    
    def timespan(self, par):
        return numpy.linspace(0, par['stop'], par['stop']*1)

    def set_data(self, t, y, u):
        self.t = t
        self.y = y
        self.u = u
        
        plt.plot(t, y, label="Real/Loaded")
        plt.legend()            
        plt.show()
        return None
    
    def get_data(self):
        plt.plot(self.t, self.y, label="Real/Loaded")
        return self.t, self.y, self.u
    
    def set_results(self, yr, dur, coeff, div):
        self.yr = yr
        self.dur = dur
        self.coeff = coeff
        self.div = div
        return None
    
    def get_results(self):
        return self.yr, self.dur, self.coeff, self.div
    
    def reset_history(self):
        self.history = []
        return None
    
    def get_history(self):
        plt.plot(self.history[:])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()
        print(f'Last Error: {self.history[-1]} \nLowest Error: {min(self.history)}')
        return self.history
    
# Possible future use:
# ------------------------------------------------------------------------
# l = locals()

# @staticmethod
#     def method_list(obj):
#         methods = []
#         for f in dir(obj):
#             if callable(getattr(obj, f)) and not f.startswith("_"):
#                 methods.append(f)   
#         funcs = [obj.l[f] for f in methods]
#         vars = [f.__code__.co_varnames for f in funcs]
#         return list(zip(funcs, vars))