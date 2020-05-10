from inspect import signature

import scipy
import psopy
import gamod
import tsmod
import sysid
import ssid

from ipywidgets import interact, interactive, interact_manual
from IPython.display import display
import IPython.display as dis
import ipywidgets as wid

import time
import os.path

import random
import pandas
import numpy
import control
from matplotlib import pyplot as plt

from scipy.optimize import rosen

class Estimate:
    def __init__(self):#, input, noise, technique, data=None):
#         self.input = input
#         self.noise = noise
#         self.technique = technique
#         self.data = data
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

    input_dict = {'Step': step, 
          'Rect': rect, 
          'Doublet': doublet}
    
    def load_data(self, file):
        if file == '':
            print('Please enter a file name to load.')
            return [0, 0, 0]
        else:
            df = pandas.read_csv(f'{file}.csv')
            ydata = df['Y'] - df['Y'][0]
            udata = df['U'] - df['U'][0]
            tdata = df['Time']
            return [ydata, udata, tdata]
    
    def save_data(self, d, save_name):
        t, y, u, yr, dur, coeff, div = d

        dd = {'Time': t,
              'Y': y,
              'U': u,
              'Y Estimate': yr}

        df = pandas.DataFrame(data=dd)

        if save_name == '':
            save_name = 'New_Save'

        while os.path.isfile(f'{save_name}.csv'):
            save_name = save_name + "(1)"

        df.to_csv(f'{save_name}.csv')

        dr = {'Run Time (s)': dur}

        for i in range(len(coeff)):
            if i < div:
                dr[f'Numerator {i}'] = [coeff[i]]
            else:
                dr[f'Denominator {i-div}'] = [coeff[i]]

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
    
    noise_dict = {'None': no_noise,
              'Uniform': uniform_noise,
              'Non-uniform': pseudo_noise}
    
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
    
    tech_dict = {'Differential Evolution': DE,
            'Least Squares': LS,
            'Scipy Minimize': MIN,
            'Particle Swarm': PSO,
            'Genetic Algorithm': GA,
            'Tabu Search': TS}
        
    def err(self, params, ts, ym, us, div):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=ts, U=us)
        return sum((ym - ysim)**2)

    def res(self, params, ts, ym, us, div):
        num = params[:div]
        den = params[div:]
        est_sys = control.tf(num, den)
        tsim, ysim, xsim = control.forced_response(est_sys, T=ts, U=us)
        plt.plot(tsim, ysim, '--', label="Estimation")
        return ysim
