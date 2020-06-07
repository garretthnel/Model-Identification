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
import json

from matplotlib import pyplot as plt

class Estimation:
    def __init__(self):
        self.input = {'Step': self.step, 
                      'Rect': self.rect, 
                      'Doublet': self.doublet}
        
        self.noise = {'None': self.no_noise,
                      'Uniform': self.uniform_noise,
                      'Non-uniform': self.pseudo_noise}
        
        self.tech = {#'Tabu Search': self.TS,
                     'Scipy Minimize': self.MIN,
                     'Differential Evolution': self.DE,
                     'Least Squares': self.LS,
                     'Particle Swarm': self.PSO,
                     'Genetic Algorithm': self.GA}
        
        self.description = {'input': '',
                            'noise': '',
                            'tech': ''}
        
        self.reset()
        self.set_id()
        
        self.num = [] 
        self.den = [] 
        
        return None
    
    def set_id(self, save=''):
        if save != '':
            self.current_id = f'{self.description["tech"]}_{self.description["input"]}_{self.description["noise"]}_{save}'
        else:
            self.current_id = f'{self.description["tech"]}_{self.description["input"]}_{self.description["noise"]}'
    
    def load_data(self, file):
        self.reset_history()
        if file == '':
            print('Please enter a file name to load.')
            return [0, 0, 0]
        else:
            with open(f'{file}.json', 'r') as f:
                d =  json.load(f)

            self.y = numpy.array(d['Y']) - d['Y'][0]
            self.u = numpy.array(d['U']) - d['U'][0]
            self.t = d['Time']
            
            plt.plot(self.t, self.y, label="Real/Loaded")
            plt.legend()
            plt.show()
            
            return self.y, self.u, self.t
    
    def save_data(self, save_name=''):
        
        self.set_id(save_name)
        
        for i in range(len(self.coeff)):
            if i < self.div:
                self.numest.append(self.coeff[i])
            else:
                self.denest.append(self.coeff[i])
        
        d = {"Technique": self.description['tech'],
             "Input": self.description['input'],
             "Noise": self.description['noise'],
             "Duration": self.dur,
             "Num": list(self.num),
             "Den": list(self.den),
             "Num Est": list(self.numest),
             "Den Est": list(self.denest),
             "Time": list(self.t),
             "Y": list(self.y),
             "U": list(self.u),
             "Y Est": list(self.yr),
             "History": list(self.history)}
        
        with open(f'{self.current_id}.json', 'w') as f:
            json.dump(d, f)

        return print(f'{self.current_id} data successfully saved.')
    
    def step(self, t, start=50, stop=600):
        self.description['input'] = 'Step'
        
        if t<start:
            return 0
        else:
            return 1

    def rect(self, t, start=50, drop=200, stop=600):
        self.description['input'] = 'Rect'
        
        if t>=start and t<stop:
            return 1
        else:
            return 0

    def doublet(self, t, start=50, drop=200, rise=250, stop=600):
        self.description['input'] = 'Doublet'
        
        if t>=start and t<drop:
            return 1
        elif t>=drop and t<rise:
            return -1
        else:
            return 0
    
    def uniform_noise(self, ydata, Magnitude=0.1):
        self.description['noise'] = 'Uniform'
        return [ydata[i] + random.uniform(-1,1)*Magnitude for i in range(len(ydata))]

    def pseudo_noise(self, ydata, Magnitude=0.1):
        self.description['noise'] = 'Pseudo'
        return [ydata[i] + random.randrange(start=-1,stop=1)*Magnitude for i in range(len(ydata))]

    def no_noise(self, ydata):
        self.description['noise'] = 'None'
        return ydata
    
    def create_system(self, Numerator, Denominator):
        self.reset()
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

    def res(self, params, div, plot):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=self.t, U=self.u)
        if plot:
            plt.plot(self.t, self.y, label="Real/Loaded")
            plt.plot(tsim, ysim, '--', label="Estimation")
            plt.legend()            
            plt.show()
        return ysim
    
    def response(self, system, t, u):
        return control.forced_response(system, T=t, U=u)
    
    def timespan(self, par):
        return numpy.linspace(0, par['stop'], par['stop']*1)

    def set_data(self, t, y, u, plot):
        self.t = t
        self.y = y
        self.u = u
        if plot:
            plt.plot(t, y, label="Real/Loaded")
            plt.legend()            
            plt.show()
        return None
    
    def get_data(self):
        return self.t, self.y, self.u
    
    def set_results(self, yr, dur, coeff, div):
        self.yr = yr
        self.dur = dur
        self.coeff = coeff
        self.div = div
        return None
    
    def get_results(self):
        return self.yr, self.dur, self.coeff, self.div
    
    def reset(self):
        self.history = []
        self.numest = []
        self.denest = []
        return None
    
    def get_history(self):
        plt.plot(self.history[:])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()
        print(f'Last Error: {self.history[-1]} \nLowest Error: {min(self.history)}')
        return self.history