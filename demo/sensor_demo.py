
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 10 11:20:41 2022

@author: mengjia, rykerfish
"""

#%% Imports
import pandas as pd
import numpy as np
import sys
utility_dir = '../'
sys.path.insert(0, utility_dir)
from utilities import wind_synthesizer
bin_dir = '../bin' # by default, makefile stores the .so file here. needs to be on the python path to get imported.
sys.path.insert(0, bin_dir)
from GaussianPuff import GaussianPuff as sensor_puff

# for plotting
import matplotlib.pylab as plt


# Load in data
data_dir = '../data/'

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

############### all of the above is code that just reads in some experimental wind data ##################

########################### sensor demo ############################
# IMPORTANT: the wind data is on 1min resolution, so obs_dt = 60 seconds
# the wind data gets resampled to sim_dt when the constructor for the python code is called.

# set simulation parameters
# IMPORTANT: obs_dt must be a positive integer multiple of sim_dt, and both sim_dt and puff_dt must be integers
obs_dt, sim_dt, puff_dt = 60, 1, 1

# start and end times at minute resolution. Needs to be in the local timezone of where we're simulating
# e.g. if we're simulating a site in England, it needs to be in UTC.
# if we're simulating a site in Colorado, it should be in MST/MDT
start = pd.to_datetime('2022-03-03 10:22:00')
end = pd.to_datetime('2022-03-03 11:52:00')

## extract wind data corresponding to start and end times
idx_0 = pd.Index(time_stamp_wind).get_indexer([start], method='nearest')[0]
idx_end = pd.Index(time_stamp_wind).get_indexer([end], method='nearest')[0]
wind_speeds = ws_syn[idx_0 : idx_end+1]
wind_directions = wd_syn[idx_0 : idx_end+1]


# emission source
source_coordinates = [[488163.338444176, 4493892.53205817, 2.0]] # format is [[x0,y0,z0]] in [m]. needs to be nested list for compatability with multi source (coming soon)
emission_rate = [3.19] # emission rate for the single source above, [kg/hr]

# sensors on the site. it is assumed that these encase the source coordinates.
sensor_coordinates = [[488164.98285821447, 4493931.649887275, 2.4],
    [488198.08502694493, 4493932.618594243, 2.4],
    [488226.9012860443, 4493887.916890612, 2.4],
    [488204.9825329503, 4493858.769131294, 2.4],
    [488172.4989330686, 4493858.565324413, 2.4],
    [488136.3904409793, 4493861.530987777, 2.4],
    [488106.145508258, 4493896.167438727, 2.4],
    [488133.15254321764, 4493932.355431944, 2.4]]

sp = sensor_puff(obs_dt=obs_dt, sim_dt=sim_dt, puff_dt=puff_dt,
                 simulation_start=start, simulation_end=end,
                 source_coordinates=source_coordinates, emission_rates=emission_rate,
                 wind_speeds=wind_speeds, wind_directions=wind_directions,
                 using_sensors=True, sensor_coordinates=sensor_coordinates, 
                 quiet=True # change to false for progress information
)

sp.simulate()

#%% plotting
t, n_sensors = np.shape(sp.ch4_obs) # (time, sensors)
sensor_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

fig, ax = plt.subplots(2, 4, figsize=(10,10))
m = sp.ch4_obs.max()
fig.supxlabel("Time from emission start (minutes)")
fig.supylabel("Methane concentration (ppm)")

for i in range(0,n_sensors):

    if i < 4:
        row = 0
        col = i
    else:
        row = 1
        col = i - 4

    times = np.arange(0, t)
    
    sensor_ch4 = sp.ch4_obs[:,i]

    ax[row][col].plot(times, sensor_ch4)
    ax[row][col].set_ylim(-1,m+2)
    ax[row][col].set_title(sensor_names[i])


fig.savefig("demo_sensors.png", format="png", dpi=500, bbox_inches="tight")
