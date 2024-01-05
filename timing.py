
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
utility_dir = '.'
sys.path.insert(0, utility_dir)
from utilities import wind_synthesizer
bin_dir = './bin'
sys.path.insert(0, bin_dir)
from GaussianPuff import GaussianPuff as GP

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

# Load in data
data_dir = './tests/test_data/'

# 1-minute resolution wind data
df_ws_1min = pd.read_csv(data_dir + 'df_ws_1min_METEC_ADET.csv') 
df_wd_1min = pd.read_csv(data_dir + 'df_wd_1min_METEC_ADET.csv')
df_ws_1min['time_stamp.mountain'] = pd.to_datetime(df_ws_1min['time_stamp.mountain'])
df_wd_1min['time_stamp.mountain'] = pd.to_datetime(df_wd_1min['time_stamp.mountain'])


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

def runSensorTest(exp_start, t_0, t_end, 
            wind_speeds, wind_directions, 
            obs_dt, sim_dt, puff_dt,
            sensor_coordinates, source_coordinates, 
            emission_rate, puff_duration, 
            unsafe=False
            ):

    sensor_puff = GP(obs_dt, sim_dt, puff_dt,
                t_0, t_end,
                source_coordinates, emission_rate,
                wind_speeds, wind_directions, 
                using_sensors=True,
                sensor_coordinates=sensor_coordinates,
                quiet=True,
                puff_duration=puff_duration, unsafe=unsafe
    )

    start = time.time()
    ch4 = sensor_puff.simulate()
    end = time.time()
    print("Runtime: ", end-start)

def timeRun(f, t_0, t_end, 
            wind_speeds, wind_directions, 
            obs_dt, sim_dt, puff_dt,
            nx, ny, nz, 
            source_coordinates, emission_rate, grid_coords, puff_duration,
            unsafe = False
            ):

    grid_puff = GP(obs_dt, sim_dt, puff_dt,
                t_0, t_end,
                source_coordinates, emission_rate,
                wind_speeds, wind_directions, 
                grid_coordinates=grid_coords,
                using_sensors=False,
                nx=nx, ny=ny, nz=nz,
                quiet=True, unsafe=unsafe,
                puff_duration=puff_duration,
    )

    print("running n=" + str(nx))
    start = time.time()
    ch4 = grid_puff.simulate()
    end = time.time()
    print("Runtime: ", end-start)
    f.write("%s \t %f\n" % (nx, end-start))

def case1():
    
    puff_duration = 1200 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
    grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]

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


    start_str = start_1

    start_time_str = start_str.replace(" ", "-").replace(":", "-")
    timing_dir = "./timings/"
    f_name = timing_dir + start_time_str + ".txt"
    print(f_name)
    f = open(f_name, "a")
    f.write("----------------------------------\n")
    f.write("EXPERIMENT START:" +  start_str + "\n")
    f.write("n \t runtime (s)\n")

    N = np.arange(5, 105, 5)
    for n in N:
      x_num = n
      y_num = n
      z_num = n
    
      timeRun(f, t_0, t_end, wind_speeds, wind_directions, 
                      obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                      source_coordinates, emission_rate, grid_coords, puff_duration)
      
def case2():
    
    puff_duration = 1200 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
    grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]

    source_coordinates = [[488206.3525776105, 4493911.77819326, 2.0]]
    emission_rate = [0.8436203738042646]

    exp_start = pd.to_datetime(start_2)
    exp_end = pd.to_datetime(end_2)

    t_0 = exp_start.floor('T')
    t_end = exp_end.floor('T')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]


    start_str = start_2

    start_time_str = start_str.replace(" ", "-").replace(":", "-")
    timing_dir = "./timings/"
    f_name = timing_dir + "new_" + start_time_str + ".txt"
    print(f_name)
    f = open(f_name, "a")
    f.write("----------------------------------\n")
    f.write("EXPERIMENT START:" +  start_str + "\n")
    f.write("n \t runtime (s)\n")

    N = np.arange(85, 105, 5)
    for n in N:
      x_num = n
      y_num = n
      z_num = n
    
      timeRun(f, t_0, t_end, wind_speeds, wind_directions, 
                      obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                      source_coordinates, emission_rate, grid_coords, puff_duration)
    

def sensor_case1():
    puff_duration = 1080 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]

    ################ TEST 1 ######################
    # uses real wind data with fabricated emission sources

    source_coordinates = [[10.5, 23, 4.5]]
    emission_rate = [1.953021587640098]
    sensor_coordinates = [[0, 1, 3.5], 
                            [12.3, 4.9, 8],
                            [-11, -4, 2],
                            [-20, 5, 2],
                            [14, -7, 4]]
    

    exp_start = pd.to_datetime(start_1)
    exp_end = pd.to_datetime(end_1)

    t_0 = exp_start.floor('T')
    t_end = exp_end.floor('T')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    runSensorTest(start_1, t_0, t_end,
                           wind_speeds, wind_directions, 
                           obs_dt, sim_dt, puff_dt,
                           sensor_coordinates, source_coordinates, 
                           emission_rate, puff_duration)


# case1()
case2()
