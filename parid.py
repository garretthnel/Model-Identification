import scipy
import psopy
import gamod
import tsmod
import sysid
import ssid

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
        return None
        
#     class Inputs:
#         @staticmethod
    def step(self, t, start, stop):
        if t<start:
            return 0
        else:
            return 1

    def rect(self, t, start, drop, stop):
        if t>=start and t<stop:
            return 1
        else:
            return 0

    def doublet(self, t, start, drop, rise, stop):
        if t>=start and t<drop:
            return 1
        elif t>=drop and t<rise:
            return -1
        else:
            return 0
    
    def load_data(self, file):
        if file == '':
            print('Please enter a file name to load.')
            return [0, 0, 0]
        else:
            df = pandas.read_csv(f'{file}.csv')
            self.y = df['Y'] - df['Y'][0]
            self.u = df['U'] - df['U'][0]
            self.t = df['Time']
            return self.y, self.u, self.t
    
    def save_data(self, save_name):

        dd = {'Time': self.t,
              'Y': self.y,
              'U': self.u,
              'Y Estimate': self.yr}

        df = pandas.DataFrame(data=dd)

        if save_name == '':
            save_name = 'New_Save'

        while os.path.isfile(f'{save_name}.csv'):
            save_name = save_name + "(1)"

        df.to_csv(f'{save_name}.csv')

        dr = {'Run Time (s)': self.dur}

        for i in range(len(coeff)):
            if i < self.div:
                dr[f'Numerator {i}'] = [self.coeff[i]]
            else:
                dr[f'Denominator {i-self.div}'] = [self.coeff[i]]

        dfr = pandas.DataFrame(data=dr)
        dfr.to_csv(f'{save_name}_results.csv')

        return print(f'{save_name} successfully saved.')
    
#     class Noise:
    def uniform_noise(self, ydata, Magnitude):
        return [ydata[i] + random.uniform(-1,1)*Magnitude for i in range(len(ydata))]

    def pseudo_noise(self, ydata, Magnitude):
        return [ydata[i] + random.randrange(start=-1,stop=1)*Magnitude for i in range(len(ydata))]

    def no_noise(self, ydata):
        return ydata
    
    def create_system(self, Numerator, Denominator):
        return control.tf(Numerator, Denominator)
        
#     class Techniques:
    def DE(self, function, Bounds):
        return scipy.optimize.differential_evolution(function, Bounds).x

    def LS(self, function, Initial_Guess):
        return scipy.optimize.least_squares(function, [i[0] for i in Initial_Guess]).x

    def MIN(self, function, Initial_Guess):
        return scipy.optimize.minimize(function, [i[0] for i in Initial_Guess]).x

    def PSO(self, function, Initial_Bounds):
        return psopy.minimize(function, numpy.array([numpy.random.uniform(*b, 10) for b in Initial_Bounds]).T).x

    def GA(self, function, Lengths):
        print(Lengths)
        return gamod.genetic_algorithm(function, [int(i[0]) for i in Lengths])

    def TS(self, function, Bounds):
        return tsmod.tabu_search(function, Bounds)
        
    def err(self, params, div):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=self.t, U=self.u)
        return sum((self.y - ysim)**2)

    def res(self, params, div):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=self.t, U=self.u)
        plt.plot(tsim, ysim, '--', label="Estimation")
        return ysim

    def set_data(self, t, y, u):
        self.t = t
        self.y = y
        self.u = u
        return None
    
    def get_data(self):
        return self.t, self.y, self.u
    
    load_state = False
    
    def set_results(self, yr, dur, coeff, div):
        self.yr = yr
        self.dur = dur
        self.coeff = coeff
        self.div = div
        return None
    
    def get_results(self):
        return self.yr, self.dur, self.coeff, self.div
    
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