
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


def runTest(exp_start, t_0, t_end, 
            wind_speeds, wind_directions, 
            obs_dt, sim_dt, puff_dt,
            nx, ny, nz, 
            grid_coords,
            puff_duration
            ):

    grid_puff = GP(obs_dt, sim_dt, puff_dt,
                t_0, t_end,
                source_coordinates, emission_rate,
                wind_speeds, wind_directions, 
                grid_coordinates=grid_coords,
                using_sensors=False,
                nx=nx, ny=ny, nz=nz,
                quiet=True,
                puff_duration=puff_duration,
    )

    ch4 = grid_puff.simulate()

    # compare to ground truth, generated using original code
    test_data_dir = "./test_data/"
    start_time_str = exp_start.replace(" ", "-").replace(":", "-")
    filename = test_data_dir + "ch4-n-" + str(grid_puff.N_points) + "-sim-" + str(sim_dt) + "-puff-" + str(puff_dt) + "-exp-" + start_time_str + ".csv"
    ch4_old = np.loadtxt(filename, delimiter=",")

    return check_test(ch4_old, ch4)

def check_test(ch4_old, ch4):

    # rewrite the test here and return whether it passed or not
    passed = True
    tol = 10e-6 # float32 precision is what the code originally used, so this is slightly larger than that
    # stop one step short of end: original code doesn't actually produce results for final time, so skip it
    for t in range(0, len(ch4_old)-1):

        if np.linalg.norm(ch4_old[t]) < 10e-3: # ppb measurements are so small we don't care about relative error
            norm = abs(np.linalg.norm(ch4_old[t].ravel() - ch4[t].ravel()))
        else:
            norm = abs(np.linalg.norm(ch4_old[t].ravel() - ch4[t].ravel())) / (np.linalg.norm(ch4_old[t].ravel()) + tol)

        if np.isnan(norm):
            print(f"ERROR: NAN present in vectorized ch4 array at time {t}")
            if passed:
                passed = False
        if norm > tol: # doesn't work if there are NAN's
            print(f"ERROR: Difference between vectorized version and original version is greater than {tol}")

            if passed:
                passed = False

    return passed

# select experiments
# source: 4T-11
start_1 = '2022-02-22 01:33:22'
end_1 = '2022-02-22 03:33:23'

# source: 5S-27
start_2 = '2022-02-26 21:36:00'
end_2 = '2022-02-26 23:07:00'

# source: 4W-47
start_3 = '2022-04-27 03:49:09'
end_3 = '2022-04-27 08:04:09'

num_tests = 0
tests_passed = 0
tests_failed = 0
failed_tests = []

################## GENERAL TESTS ###################

x_num = 20
y_num = 20
z_num = 20

puff_duration = 1080 # used in the original python code

obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]

################ TEST 1 ######################
num_tests += 1

source_coordinates = [[488163.3384441765, 4493892.532058168, 4.5]]
emission_rate = [1.953021587640098]

exp_start = pd.to_datetime(start_1)
exp_end = pd.to_datetime(end_1)

t_0 = exp_start.floor('T')
t_end = exp_end.floor('T')

idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
wind_speeds = ws_syn[idx_0 : idx_end+1]
wind_directions = wd_syn[idx_0 : idx_end+1]

print("RUNNING TEST 1")
passed = runTest(start_1, t_0, t_end, wind_speeds, wind_directions, 
                 obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, grid_coords, puff_duration)

if not passed:
    print("FAILED TEST ONE")
    tests_failed += 1
else:
    print ("TEST ONE PASSED")
    tests_passed += 1



# elif i == 1:
#     source_coordinates = [[488206.3525776105, 4493911.77819326, 2.0]]
#     emission_rate = [0.8436203738042646]
# elif i == 2:
#     source_coordinates = [[488124.41821990383, 4493915.016403197, 2.0]]
#     emission_rate = [0.5917636636467585]

# grid_puff = GP(obs_dt, sim_dt, puff_dt,
#                 t_0, t_end,
#                 source_coordinates, emission_rate,
#                 wind_speeds, wind_directions, 
#                 grid_coordinates=grid_coords,
#                 using_sensors=False,
#                 nx=x_num, ny=y_num, nz=z_num,
#                 quiet=False,
#                 puff_duration=puff_duration
# )

# start = time.time()
# ch4 = grid_puff.simulate()
# end = time.time()

# runtime = end-start
# print("runtime: ", runtime)
# print(f"simulation length (real time): {grid_puff.n_obs} minutes")

# # compare to ground truth, generated using original code
# test_data_dir = "./test_data/"
# start_time_str = exp_start_time.replace(" ", "-").replace(":", "-")
# filename = test_data_dir + "ch4-n-" + str(grid_puff.N_points) + "-sim-" + str(sim_dt) + "-puff-" + str(puff_dt) + "-exp-" + start_time_str + ".csv"
# ch4 = np.loadtxt(filename, delimiter=",")


# print("NUMER OF TESTS:", num_tests)
# print("TESTS PASSED:", tests_passed)
# print("TESTS FAILED:", tests_failed)
# if(tests_failed > 0):
#     print("Failed on test numbers", failed_tests)
