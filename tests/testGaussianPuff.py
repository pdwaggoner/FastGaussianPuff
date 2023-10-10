
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: mengjia, rykerfish
"""

#%% Imports
import pandas as pd
import numpy as np
import time
# import utility functions
import sys
utility_dir = '../'
sys.path.insert(0, utility_dir)
from utilities import wind_synthesizer
bin_dir = '../bin'
sys.path.insert(0, bin_dir)
from GaussianPuff import GaussianPuff as GP



#%% 
# Load in data
data_dir = '../data/demo_data/'

# 1-minute resolution wind data
df_ws_1min = pd.read_csv(data_dir + 'df_ws_1min_METEC_ADET.csv') 
df_wd_1min = pd.read_csv(data_dir + 'df_wd_1min_METEC_ADET.csv')
df_ws_1min['time_stamp.mountain'] = pd.to_datetime(df_ws_1min['time_stamp.mountain'])
df_wd_1min['time_stamp.mountain'] = pd.to_datetime(df_wd_1min['time_stamp.mountain'])


# experiment data
df_experiment = pd.read_csv(data_dir + 'df_exp_METEC_ADET.csv')
df_experiment['start_time.mountain'] = pd.to_datetime(df_experiment['start_time.mountain'])
df_experiment['end_time.mountain'] = pd.to_datetime(df_experiment['end_time.mountain'])

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

# synethize wind data- combines wind data from multiple sensors into one timeseries
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

num_tests = 0
tests_passed = 0
tests_failed = 0
failed_tests = []

for i in range(0, len(starts)):

    num_tests += 1

    exp_start_time = starts[i]
    exp_end_time = ends[i]

    df_experiment_sub = \
    df_experiment.loc[ ( df_experiment['start_time.mountain'] >= pd.to_datetime(exp_start_time) ) &
                    ( df_experiment['start_time.mountain'] <= pd.to_datetime(exp_end_time) ) ].reset_index(drop = True)

    # set simulation parameters
    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]

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
        wind_speeds = ws_syn[idx_0 : idx_end+1]
        wind_directions = wd_syn[idx_0 : idx_end+1]
        
        x_num = 20
        y_num = 20
        z_num = 20

        puff_duration = 1080 # used in the original python code


        grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]
        if i == 0:
            source_coordinates = [[488163.3384441765, 4493892.532058168, 4.5]]
            emission_rate = [1.953021587640098]
        elif i == 1:
            source_coordinates = [[488206.3525776105, 4493911.77819326, 2.0]]
            emission_rate = [0.8436203738042646]
        elif i == 2:
            source_coordinates = [[488124.41821990383, 4493915.016403197, 2.0]]
            emission_rate = [0.5917636636467585]

        grid_puff = GP(obs_dt, sim_dt, puff_dt,
                        t_0, t_end,
                        source_coordinates, emission_rate,
                        wind_speeds, wind_directions, 
                        grid_coordinates=grid_coords,
                        using_sensors=False,
                        nx=x_num, ny=y_num, nz=z_num,
                        quiet=False,
                        puff_duration=puff_duration
        )

        start = time.time()
        ch4_vect = grid_puff.simulate()
        end = time.time()

        runtime = end-start
        print("runtime: ", runtime)
        print(f"simulation length (real time): {grid_puff.n_obs} minutes")

        # compare to ground truth, generated using original code
        test_data_dir = "./test_data/"
        start_time_str = exp_start_time.replace(" ", "-").replace(":", "-")
        filename = test_data_dir + "ch4-n-" + str(grid_puff.N_points) + "-sim-" + str(sim_dt) + "-puff-" + str(puff_dt) + "-exp-" + start_time_str + ".csv"
        ch4 = np.loadtxt(filename, delimiter=",")

        passed = True
        tol = 10e-6 # float32 precision is what the code originally used, so this is slightly larger than that
        # stop one step short of end: original code doesn't actually produce results for final time, so skip it
        for t in range(0, len(ch4)-1):

            if np.linalg.norm(ch4[t]) < 10e-3: # ppb measurements are so small we don't care about relative error
                norm = abs(np.linalg.norm(ch4[t].ravel() - ch4_vect[t].ravel()))
            else:
                norm = abs(np.linalg.norm(ch4[t].ravel() - ch4_vect[t].ravel())) / (np.linalg.norm(ch4[t].ravel()) + tol)

            if np.shape(ch4_vect[t]) != grid_puff.grid_dims:
                print(f"ERROR: CH4 ARRAY AT TIME {t} IS WRONG SHAPE")
                tests_failed += 1

            if np.isnan(norm):
                print(f"ERROR: NAN present in vectorized ch4 array at time {t}")
                if not passed:
                    passed = False
            if norm > tol: # doesn't work if there are NAN's
                print(f"ERROR: Difference between vectorized version and original version is greater than {tol}")

                if passed:
                    passed = False
                print("TIME:", t)
                print("RELATIVE NORM", norm)

        if not passed:
            print(f"Test {i+1} failed")
            tests_failed += 1
            failed_tests.append(i+1)

        else:
            print(f"Test {i+1} passed")
            tests_passed += 1

print("NUMER OF TESTS:", num_tests)
print("TESTS PASSED:", tests_passed)
print("TESTS FAILED:", tests_failed)
if(tests_failed > 0):
    print("Failed on test numbers", failed_tests)
