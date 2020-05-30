from ipywidgets import interact, interactive, interact_manual
from IPython.display import display
import IPython.display as dis
import ipywidgets as wid

import time

import importlib

import parid
import numpy

est = parid.Estimation()

def sim(Input='Step', Noise="None", Technique='Genetic Algorithm'):

    Load_Name = wid.Text(value='', description='Load Name', disabled=False)
    run_load = wid.Button(description="Load", disabled=False)
    
    def run_load_clicked(b):
        with output:
            dis.clear_output(True)
            est.load_data(Load_Name.value)
            
    run_load.on_click(run_load_clicked)
    
    u = est.input[Input]
    u_sig = u.__code__.co_varnames[2:]
    c_input_params = {}
    c_input_wid = {}
    for var in u_sig:
        c_input_wid[var] = wid.IntSlider(value=1, min=0, max=600, description=var, disabled=False)
        display(c_input_wid[var])
        
    n = est.noise[Noise]
    n_sig = n.__code__.co_varnames[2:]
    c_noise_params = {}
    c_noise_wid = {}
    for var in n_sig:
        c_noise_wid[var] = wid.FloatSlider(value=0, min=0, max=0.5, step=0.01, description=var, disabled=False)
        display(c_noise_wid[var])
    
    sys = est.create_system
    sys_sig = sys.__code__.co_varnames[1:]
    c_sys_params = {}
    c_sys_wid = {}
    for var in sys_sig:
        c_sys_wid[var] = wid.Text(value='1', description=var, disabled=False)
        display(c_sys_wid[var])
        
    tech = est.tech[Technique]
    tech_sig = tech.__code__.co_varnames[2:]
    for names in tech_sig:
        display(names+" (inputs separated by a semicolon)")
    
    c_tech_wid = {}
    for var1 in tech_sig:
        for var2 in sys_sig:
            var = str(var1) + str(var2)
            c_tech_wid[var] = wid.Text(value='1', description=var2, disabled=False)
            display(c_tech_wid[var])
    
    Save_Name = wid.Text(value='', description='Save Name', disabled=False)
    display(Save_Name)
    display(Load_Name)
    
    run_sim = wid.Button(description='Simulate')
    run_est = wid.Button(description='Estimate')
    
    run_state = False
    
    output = wid.Output()

    def run_sim_clicked(b):
        with output:
            dis.clear_output(True)
            
            for var in u_sig:
                c_input_params[var] = c_input_wid[var].value
                
            for var in n_sig:
                c_noise_params[var] = c_noise_wid[var].value
            
            for var in sys_sig:
                c_sys_params[var] = [float(i) for i in c_sys_wid[var].value.split(';')]
            
            ts = est.timespan(c_input_params)
            us = [u(t, *c_input_params.values()) for t in ts]
            
            system = est.create_system(*c_sys_params.values()) 
            tm, y1, xm = est.response(system, ts, us)
            
            ym = n(y1, *c_noise_params.values())
            est.set_data(ts, ym, us)
        
    def run_est_clicked(b):
        with output:
            dis.clear_output(True)
            arr_tech_params = []
            orders = []
            
            for o, var in enumerate(c_tech_wid):
                orders.append(len(c_tech_wid[var].value.split(';')))
                for k in range(orders[o]):
                    arr_tech_params.append([float(j) for j in [i.split(',') for i in c_tech_wid[var].value.split(';')][k]]) 
            
            div = orders[0]
            
            def err_wrap(params):
                return est.err(params, div)
        
            def res_wrap(params):
                return est.res(params, div)
            
            print('Running...')
            st = time.time()
            cs = tech(err_wrap, arr_tech_params)
            coeff = numpy.array(cs)/min([abs(i) for i in cs if abs(i) > 1e-3])
            et = time.time()
            print('Done.')
            
            dur = et-st
            print("Parameters estimated as {} in {} seconds".format(coeff, dur))
            
            ts, ym, _ = est.get_data()
            
            yres = res_wrap(coeff)
            
            est.set_results(yres, dur, coeff, div)
            est.get_history()
            
            run_state = True   

#     def run_all_clicked(b):
#         with output:
#             dis.clear_output(True)
#             for systems in system_list:
#                 for keys, tech in est.tech.items():
#                     tech_sig = tech.__code__.co_varnames[2:]
#                     for names in tech_sig:
#                         if 'Bounds' in names:
#                             print('B')
#                         elif 'Guess' in names:
#                             print('G')
#                         elif 'Lengths' in names:
#                             print('L')
                    
#     run_all = wid.Button(description='Run all')
#     display(run_all)
#     run_all.on_click(run_all_clicked)
    
    display(run_sim)
    display(run_est, output)
    
    run_sim.on_click(run_sim_clicked)
    run_est.on_click(run_est_clicked)
    
    run_save = wid.Button(description="Save", disabled=run_state)
    
    display(run_save)
    display(run_load)
    
    def run_save_clicked(b):
        with output:
            est.save_data(Save_Name.value)
            
    run_save.on_click(run_save_clicked)
    
    
system_list = []
    
def gui():
    return interact(sim, Input=[keys for keys, dict in est.input.items()], Noise=[keys for keys, dict in est.noise.items()], Technique=[keys for keys, dict in est.tech.items()])