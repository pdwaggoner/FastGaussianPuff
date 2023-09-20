
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
from GaussianPuff import GAUSSIANPUFF
bin_dir = './bin'
sys.path.insert(0, bin_dir)
from vectorizedGaussianPuff import vectorizedGAUSSIANPUFF



#%% 
# Load in data
# specify paths
data_dir = './data/clean_data/'
simulation_data_dir = './puff_results/'
plot_save_dir = './puff_plots/'

# 1-minute-resolution cm data
df_ch4_1min = pd.read_csv(data_dir + 'df_ch4_1min_METEC_ADET.csv',) 
df_ws_1min = pd.read_csv(data_dir + 'df_ws_1min_METEC_ADET.csv') 
df_wd_1min = pd.read_csv(data_dir + 'df_wd_1min_METEC_ADET.csv')
df_ch4_1min['time_stamp.mountain'] = pd.to_datetime(df_ch4_1min['time_stamp.mountain'])
df_ws_1min['time_stamp.mountain'] = pd.to_datetime(df_ws_1min['time_stamp.mountain'])
df_wd_1min['time_stamp.mountain'] = pd.to_datetime(df_wd_1min['time_stamp.mountain'])


# experiment data
df_experiment = pd.read_csv(data_dir + 'df_exp_METEC_ADET.csv')
df_experiment['start_time.mountain'] = pd.to_datetime(df_experiment['start_time.mountain'])
df_experiment['end_time.mountain'] = pd.to_datetime(df_experiment['end_time.mountain'])

# location data
df_source_locs = pd.read_csv(data_dir + 'df_source_locs_METEC_ADET.csv')
df_sensor_locs = pd.read_csv(data_dir + 'df_sensor_locs_METEC_ADET.csv')

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
start_1 = '2022-02-22 01:33:22'
end_1 = '2022-02-22 03:33:23'

# source: 5S-27
start_2 = '2022-02-26 21:36:00'
end_2 = '2022-02-26 23:07:00'

# source: 4W-47
start_3 = '2022-04-27 03:48:00'
end_3 = '2022-04-27 08:05:00'

starts = [start_1, start_2, start_3]
ends = [end_1, end_2, end_3]

num_tests = len(starts)
tests_passed = 0
tests_failed = 0
failed_tests = []

for i in range(0, len(starts)):

    exp_start_time = starts[i]
    exp_end_time = ends[i]

    df_experiment_sub = \
    df_experiment.loc[ ( df_experiment['start_time.mountain'] >= pd.to_datetime(exp_start_time) ) &
                    ( df_experiment['start_time.mountain'] <= pd.to_datetime(exp_end_time) ) ].reset_index(drop = True)

    # set simulation parameters
    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
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
        
        # initialize GAUSSIANPUFF class objects
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

        # lie to the old code to make it run on a grid instead of sensors
        # N_points = vect_puff.N_points
        # puff.x_sensor = vect_puff.X.ravel()
        # puff.y_sensor = vect_puff.Y.ravel()
        # puff.z_sensor = vect_puff.Z.ravel()
        # puff.N_sensor = N_points
        # puff.ch4_sim = np.zeros((puff.N_t_sim, vect_puff.N_points))
        # puff.ch4_sim_res = np.zeros((puff.N_t_obs, vect_puff.N_points))

        vect_start = time.time()
        ch4_vect = vect_puff.simulation()
        vect_end = time.time()

        vect_time = vect_end-vect_start
        print("new runtime: ", vect_time)
        print(f"simulation length: {len(vect_puff.time_stamps)} minutes")

        # IF RUNNING TESTS FREQUENTLY: Save the array created using the original GP
        # model using np.save and load it in using np.load (as below)

        # puff_start = time.time()
        # ch4 = puff.simulation()
        # puff_end = time.time()

        # puff_time = puff_end-puff_start
        # print("old runtime: ", puff_end-puff_start)

        test_data_dir = "./data/test_data/"
        start_time_str = exp_start_time.replace(" ", "-").replace(":", "-")
        filename = test_data_dir + "ch4-n-" + str(vect_puff.N_points) + "-exp-" + start_time_str + ".csv"
        # np.savetxt(filename, ch4, delimiter=",")
        ch4 = np.loadtxt(filename, delimiter=",")

        passed = True
        tol = 10e-6 # float32 precision is what the code originally used, so this is slightly larger than that
        for t in range(0, len(ch4)):

            if np.linalg.norm(ch4[t]) < 10e-3: # ppb measurements are so small we don't care about relative error
                norm = abs(np.linalg.norm(ch4[t].ravel() - ch4_vect[t].ravel()))
            else:
                norm = abs(np.linalg.norm(ch4[t].ravel() - ch4_vect[t].ravel())) / (np.linalg.norm(ch4[t].ravel()) + tol)

            # if np.shape(ch4_vect[t]) != vect_puff.grid_dims:
            #     print(f"ERROR: CH4 ARRAY AT TIME {t} IS WRONG SHAPE")
            #     tests_failed += 1

            # if np.isnan(norm):
            #     print(f"ERROR: NAN present in vectorized ch4 array at time {t}")
            #     if not passed:
            #         passed = False
            if norm > tol: # doesn't work if there are NAN's
                print(f"ERROR: Difference between vectorized version and original version is greater than {tol}")

                if passed:
                    passed = False
                print("TIME:", t)
                print("RELATIVE NORM", norm)

        if not passed:
            print(f"Test {i+1} failed")
            tests_failed += 1

        else:
            print(f"Test {i+1} passed")
            tests_passed += 1

print("NUMER OF TESTS:", num_tests)
print("TESTS PASSED:", tests_passed)
print("TESTS FAILED:", tests_failed)
if(tests_failed > 0):
    print("Failed on test numbers", failed_tests)
