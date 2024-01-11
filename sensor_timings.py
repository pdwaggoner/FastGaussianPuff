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

def runSensorSim(fd, exp_index, source_id, t_0, t_end, 
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
    sensor_puff.simulate()
    end = time.time()

    fd.write("%d,%s,%s,%s,%f\n" % (exp_index, source_id, exp_start, exp_end, end-start))

# Load in data
data_dir = './tests/test_data/'

# 1-minute resolution wind data
df_ws_1min = pd.read_csv(data_dir + 'df_ws_1min_METEC_ADET.csv') 
df_wd_1min = pd.read_csv(data_dir + 'df_wd_1min_METEC_ADET.csv')
df_ws_1min['time_stamp.mountain'] = pd.to_datetime(df_ws_1min['time_stamp.mountain'])
df_wd_1min['time_stamp.mountain'] = pd.to_datetime(df_wd_1min['time_stamp.mountain'])

# experiment data
df_experiment = pd.read_csv("./data/df_exp_METEC_ADET.csv") 
df_source = pd.read_csv("./data/df_source_locs_METEC_ADET.csv")
df_sensor = pd.read_csv("./data/df_sensor_locs_METEC_ADET.csv")

x = df_sensor["utm_easting.m"]
y = df_sensor["utm_northing.m"]
z = df_sensor["height.m"]

sensor_coords = []
for i in range(0, x.size):
    sensor_coords.append([x.values[i], y.values[i], z.values[i]])

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

#%%
f_name = "./timings/algorithm_sensor_campaign.txt"
file = open(f_name, "a")
file.write("exp_id,source_id,exp_start,exp_end,runtime\n")

for index, row in df_experiment.iterrows():
    print("exp:", index)
    exp_start = pd.to_datetime(row["start_time.mountain"])
    exp_end = pd.to_datetime(row["end_time.mountain"])
    source_id = row["name"]
    emission_rate = row["emission_rate.kg/hr"]

    exp_start = exp_start.floor('T')
    exp_end = exp_end.floor('T')

    
    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]

    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
    puff_duration = 1200

    source = df_source.loc[df_source["name"] == source_id]

    source_coords = [[source["utm_easting.m"].values[0], source["utm_northing.m"].values[0], source["height.m"].values[0]]]

    runSensorSim(file, index, source_id, exp_start, exp_end, wind_speeds, wind_directions,
                 obs_dt, sim_dt, puff_dt, sensor_coords, source_coords, [emission_rate], puff_duration)
    
