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
            
            self.load_state = True
            plt.plot(self.t, self.y, label="Real/Loaded")
            plt.legend()
            plt.show()
            
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

        for i in range(len(self.coeff)):
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
        self.coeff = scipy.optimize.differential_evolution(function, Bounds).x
        return self.coeff

    def LS(self, function, Initial_Guess):
        self.coeff = scipy.optimize.least_squares(function, [i[0] for i in Initial_Guess]).x
        return self.coeff

    def MIN(self, function, Initial_Guess):
        self.coeff = scipy.optimize.minimize(function, [i[0] for i in Initial_Guess]).x
        return self.coeff

    def PSO(self, function, Initial_Bounds):
        self.coeff = psopy.minimize(function, numpy.array([numpy.random.uniform(*b, 10) for b in Initial_Bounds]).T).x
        return self.coeff

    def GA(self, function, Lengths):
        self.coeff = gamod.genetic_algorithm(function, [int(i[0]) for i in Lengths])
        return self.coeff

    def TS(self, function, Bounds):
        self.coeff = tsmod.tabu_search(function, Bounds)
        return self.coeff
        
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
    
    load_state = False
    
    def set_results(self, yr, dur, coeff, div):
        self.yr = yr
        self.dur = dur
        self.coeff = coeff
        self.div = div
        return None
    
    def get_results(self):
        return self.yr, self.dur, self.coeff, self.div
    
#     def estimate(self, u, n, sys):
        
#         for var in u.keys():
#             input_params[var] = u[var].value

#         for var in n.keys():
#             noise_params[var] = n[var].value

#         for var in sys.keys():
#             sys_params[var] = [float(i) for i in sys[var].value.split(';')]
        
#         ts = numpy.linspace(0, input_params['stop'], input_params['stop']*1)
#         us = [u(t, *input_params.values()) for t in ts]

#         system = control.tf(*sys_params.values())
#         tm, y1, xm = control.forced_response(system, T=ts, U=us)

#         ym = n(y1, *c_noise_params.values())
#         est.set_data(ts, ym, us)

#         plt.plot(tm, ym, label="Real/Loaded")
#         plt.legend()            
#         plt.show()
    
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