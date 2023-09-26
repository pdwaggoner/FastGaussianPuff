
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 10 11:20:41 2022

@author: mengjia, rykerfish
"""

#%% Imports
import pandas as pd
import numpy as np
# import my own utility functions
import sys
utility_dir = '../'
sys.path.insert(0, utility_dir)
from utilities import wind_synthesizer
bin_dir = '../bin' # by default, makefile stores the .so file here. needs to be on the python path to get imported.
sys.path.insert(0, bin_dir)
from GaussianPuff import GaussianPuff as GP


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
start = '2022-02-22 01:33:22'
end = '2022-02-22 03:33:23'

df_experiment_sub = \
df_experiment.loc[ ( df_experiment['start_time.mountain'] >= pd.to_datetime(start) ) &
                ( df_experiment['start_time.mountain'] <= pd.to_datetime(end) ) ].reset_index(drop = True)


### all of the above is code that just reads in some wind data. 


# set simulation parameters
# IMPORTANT: obs_dt must be a positive integer multiple of sim_dt, and sim_dt must be an integer
obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]


# IMPORTANT: the wind data is on 1min resolution, so obs_dt = 60 seconds
# the wind data gets resampled to sim_dt when the constructor for the python code is called.


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

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    ## extract inputs to the puff model
    idx_0 = pd.Index(time_stamp_wind).get_indexer([t_0], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([t_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]
    
    x_num = 20
    y_num = 20
    z_num = 20

    grid_coords = [-100, -80, 0, 100, 80, 24.0] # format is (x_min, y_min, z_min, x_max, y_max, z_max) in [m]
    
    # format is [[x0,y0,z0]] in [m]. needs to be nested list for compatability with multi source (coming soon)
    source_coordinates = [[10, 20, 4.5]]
    emission_rate = [3] # emission rate for the single source above, [kg/hr]

    grid_puff = GP(wind_speeds, wind_directions, 
                                    obs_dt, sim_dt, puff_dt,
                                    start, end,
                                    source_coordinates,
                                    emission_rate,
                                    grid_coordinates=grid_coords,
                                    using_sensors=False,
                                    nx=x_num, ny=y_num, nz=z_num,
                                    quiet=False
    )

    grid_puff.simulate()

    temp = []

    # flatten the concentration array for easier handling in matlab
    for i in range(grid_puff.n_obs):
        temp.append(grid_puff.ch4_obs[i].ravel())

    np.savetxt("concentration.csv", temp)
    np.savetxt("X.csv", grid_puff.X)
    np.savetxt("Y.csv", grid_puff.Y)
    np.savetxt("Z.csv", grid_puff.Z)
