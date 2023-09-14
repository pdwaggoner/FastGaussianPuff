
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 10 11:20:41 2022

@author: mengjia
"""

#%% Imports
from platform import python_branch
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import multiprocessing as mp
import random
import numpy as np
import time
# import my own utility functions
import sys
utility_dir = './'
sys.path.insert(0, utility_dir)
from utilities import wind_synthesizer, wrapper_run_puff_simulation, combine_subexperiments
# from GaussianPuff import GAUSSIANPUFF
bin_dir = './bin'
sys.path.insert(0, bin_dir)
from vectorizedGaussianPuff import vectorizedGAUSSIANPUFF



#%% 
# Load in data
# specify paths
data_dir = '.data/clean_data/'
simulation_data_dir = './puff_results/'
plot_save_dir = './puff_plots/'

# 1-minute-resolution cm data
df_ch4_1min = pd.read_pickle(data_dir + 'df_ch4_1min_METEC_ADET.pkl') 
df_ws_1min = pd.read_pickle(data_dir + 'df_ws_1min_METEC_ADET.pkl') 
df_wd_1min = pd.read_pickle(data_dir + 'df_wd_1min_METEC_ADET.pkl') 

# experiment data
df_experiment = pd.read_pickle(data_dir + 'df_exp_METEC_ADET.pkl')

# location data
df_source_locs = pd.read_pickle(data_dir + 'df_source_locs_METEC_ADET.pkl')
df_sensor_locs = pd.read_pickle(data_dir + 'df_sensor_locs_METEC_ADET.pkl')

#%% Data processing
# Data processing
# column names used in the load in dfs
colnames = {'name' : 'name', 
            'x' : 'utm_easting.m',
            'y' : 'utm_northing.m',
            'z' : 'height.m',
            't' : 'time_stamp.mountain',
        'exp_id' : 'experiment_id', 
        'exp_t_0' : 'start_time.mountain', 
    'exp_t_end' : 'end_time.mountain', 
'emission_rate' : 'emission_rate.kg/hr'}

# ch4 data
ch4_obs_all = df_ch4_1min.loc[:, df_ch4_1min.columns!=colnames['t']].to_numpy()
time_stamp_ch4 = df_ch4_1min[colnames['t']].to_list()

# synethize wind data
if df_ws_1min.shape == df_wd_1min.shape:
    wind_syn_mode, wind_sensor = 'circular_mean', None
    ws_syn, wd_syn = wind_synthesizer(df_ws_1min, df_wd_1min, 
                                    wind_syn_mode, wind_sensor = wind_sensor,
                                    colname_t = colnames['t'])
    time_stamp_wind = df_ws_1min[colnames['t']].to_list()
else:
    raise ValueError(">>>>> df_ws and df_wd must have the same shape.") 



# select experiments

# source: 4T-11
exp_start_time = '2022-02-22 01:33:22'
exp_end_time = '2022-02-22 03:33:23'

# source: 5S-27
# exp_start_time = '2022-02-26 21:36:00'
# exp_end_time = '2022-02-26 23:07:00'

# source: 4W-47
# exp_start_time = '2022-04-27 03:48:00'
# exp_end_time = '2022-04-27 08:05:00'


df_experiment_sub = \
df_experiment.loc[(df_experiment['start_time.mountain']>= pd.to_datetime(exp_start_time)) &
                (df_experiment['start_time.mountain']<= pd.to_datetime(exp_end_time))].reset_index(drop = True)

# set simulation parameters
obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
parallel, n_core = True, 3
quiet = False 

exp_run_start_time = datetime.datetime.now() # log run time

# initialize result containers
exp_id_list, puff_list = [], []

# initialize puffs corresponding to each experiment
for index, row in df_experiment_sub.iterrows():
    exp_id, source_name, t_0, t_end, emission_rate = row[[colnames['exp_id'],
                                                        colnames['name'], 
                                                        colnames['exp_t_0'],
                                                        colnames['exp_t_end'],
                                                        colnames['emission_rate']]]
    
    ## floor t_0 and t_end to the nearest minute
    t_0 = t_0.floor('T')
    t_end = t_end.floor('T')
    
    ## extract inputs to the puff model
    idx_0 = pd.Index(time_stamp_wind).get_indexer([t_0], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([t_end], method='nearest')[0]
    time_stamps = time_stamp_wind[idx_0 : idx_end+1]
    source_names = [source_name] * len(time_stamps)
    emission_rates = [emission_rate] * len(time_stamps)
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]
    
    ## initialize GAUSSIANPUFF class objects
    # puff = GAUSSIANPUFF(time_stamps, 
    #                     source_names, emission_rates, 
    #                     wind_speeds, wind_directions,
    #                     obs_dt, sim_dt, puff_dt, 
    #                     df_sensor_locs, df_source_locs,
    #                     quiet=False,
    #                     model_id = index
    #                     )
    
    x_num = 20
    y_num = 20
    z_num = 20

    vect_puff = vectorizedGAUSSIANPUFF(time_stamps, 
                                       source_names, emission_rates, 
                                       wind_speeds, wind_directions, 
                                       obs_dt, sim_dt, puff_dt,
                                       df_sensor_locs, df_source_locs,
                                       using_sensors=False,
                                       x_num=x_num, y_num=y_num, z_num=z_num,
                                       quiet=False
    )

vect_start = time.time()
ch4_vect = vect_puff.simulation()
vect_end = time.time()

vect_time = vect_end-vect_start
print("runtime: ", vect_time)
print(f"simulation length: {len(vect_puff.time_stamps)} minutes")

test_data_dir = "./test_data/"
start_time_str = exp_start_time.replace(" ", "-").replace(":", "-")

filename = test_data_dir + "ch4_n=" + str(vect_puff.N_points) + "_" + start_time_str + "-larger-z.npy"

ch4 = np.load(filename)


passed = True
tol = 10e-6 # float32 precision is what the code originally used, so this is slightly larger than that
for t in range(0, len(ch4)):

    if np.linalg.norm(ch4[t]) < 10e-3: # ppb measurements are so small we don't care about relative error
        norm = abs(np.linalg.norm(ch4[t].ravel() - ch4_vect[t].ravel()))
    else:
        norm = abs(np.linalg.norm(ch4[t].ravel() - ch4_vect[t].ravel())) / (np.linalg.norm(ch4[t].ravel()) + tol)

    if np.shape(ch4_vect[t]) != vect_puff.grid_dims:
        print(f"ERROR: CH4 ARRAY AT TIME {t} IS WRONG SHAPE")
        passed = False

    if np.isnan(norm):
        print(f"ERROR: NAN present in vectorized ch4 array at time {t}")
        passed = False
    elif norm > tol: # doesn't work if there are NAN's, has to stay in the elif
        print(f"ERROR: Difference between vectorized version and original version is greater than {tol}")
        passed = False
        print("TIME:", t)
        # print(ch4[t])
        # print(ch4_vect[t].ravel())
        # print("norm:", np.linalg.norm(ch4[t]))
        # print("vect_norm:", np.linalg.norm(ch4_vect[t]))
        print("RELATIVE NORM", norm)

if not passed:
    print("Test failed")
else:
    print("Test passed")
# %%
