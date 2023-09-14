#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:59:36 2022

@author: mengjia
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime


class GAUSSIANPUFF:
    def __init__(self, 
                 time_stamps, source_names, emission_rates, wind_speeds, wind_directions,
                 obs_dt, sim_dt, puff_dt, 
                 df_sensor_locations, df_source_locations,
                 puff_duration = 1080,
                 colnames = {'name' : 'name', 
                                'x' : 'utm_easting.m',
                                'y' : 'utm_northing.m',
                                'z' : 'height.m',
                                't' : 'time_stamp.mountain'},
                 conversion_factor = 1e6*1.524,
                 non_emission_char = 'None',
                 quiet = False,
                 ch4_obs = None,
                 model_id = None):
        
        '''
        Inputs: 
            time_stamps (list of pd.DataTime values): 
                time stamps of the observed continuous monitoing (cm) data
            source_names (list of str): 
                true emission source name at each time stamp
            non_emission_char (str): 
                a character denoting non-emission case in source_names
            emission_rates [kg/hr] (list of floats): 
                true emission rate at each time stamp
            wind_speeds [m/s] (list of floats): 
                wind speed at each time stamp
            wind_directions [degree] (list of floats): 
                wind direction at each time stamp, 
                following the conventional definition: 
                0 -> wind blowing from North, 90 -> E, 180 -> S, 270 -> W
            obs_dt [s] (scalar, float): 
                time interval (dt) for the observations
            sim_dt [s] (scalar, float): 
                time interval (dt) for the simulation results
            puff_dt [s] (scalar, float): 
                time interval (dt) between two successive puffs' creation, 
                usually it's set to be equal to sim_dt
            df_sensor_locations (pd.DataFrame): 
                locations of sensors, 
                columns include: name, utm_easting, utm_northing, and height
            df_source_locations (pd.DataFrame): 
                locations of sources
                columns include: name, utm_easting, utm_northing, and height
            puff_duration [s] (int):
                how many seconds a puff can 'live'; we assume a puff will fade away after a certain time
            colnames (dictionary): 
                column names used in the input dataframes
            conversion_factor (scalar, float): 
                convert from kg/m^3 to ppm, this factor is for ch4 only
            quiet (boolean): 
                if output progress information while running or not
            ch4_obs [ppm] (2D array, shape = [N_t_obs, N_sensor]): 
                the observed ch4 concentration if available
                the column order of ch4_obs must be consistent with sensor_names
            model_id (int or str):
                unique id for the current puff object, 
                useful to distinguish multiple puff objects when run code parallelly   
        '''
        
        # Initialize the configuration
        self.time_stamps = time_stamps 
        self.start_time = time_stamps[0]
        self.end_time = time_stamps[-1]
        self.N_t_obs = len(self.time_stamps) # time stamp length in observation 
        self.source_names = source_names 
        self.non_emission_char = non_emission_char 
        self.emission_rates = [x/3600 for x in emission_rates] # convert from [kg/hr] to [kg/s]
        self.wind_speeds = wind_speeds 
        self.wind_directions = wind_directions 
        self.sensor_locations = df_sensor_locations 
        self.source_locations = df_source_locations 
        self.obs_dt = obs_dt 
        self.sim_dt = sim_dt 
        self.puff_dt = puff_dt # usually the same as sim_dt
        self.time_stamps_puff_creation = \
        list(pd.date_range(start = time_stamps[0], 
                           end = time_stamps[-1], 
                           freq = str(puff_dt)+'S')) # time stamps when puffs are created
                                                       
        self.puff_duration = puff_duration
        self.colnames = colnames
        self.conversion_factor = conversion_factor 
        self.quiet = quiet
        self.ch4_obs = ch4_obs
        self.model_id = model_id
        
        # resample the input arrays by the given sim_dt
        self._resample_inputs(self.sim_dt)
        self.N_t_sim = len(self.time_stamps_res) # time stamp length of simulation data 
        
        # extract sensor location information
        self.sensor_names = df_sensor_locations[colnames['name']].to_list() # list of all sensor names 
        self.x_sensor = df_sensor_locations[colnames['x']].to_list() # list of x coordinates of all the sensors
        self.y_sensor = df_sensor_locations[colnames['y']].to_list() # y coordinates 
        self.z_sensor = df_sensor_locations[colnames['z']].to_list() # z coordinates 
        self.N_sensor = len(self.sensor_names) # number of sensors
        
        # extract source location information
        self.source_names = df_source_locations[colnames['name']].to_list() # list of all potential source names
        self.x_source = df_source_locations[colnames['x']].to_list() # list of x coordinates of all potential sources
        self.y_source = df_source_locations[colnames['y']].to_list() # y coordinates 
        self.z_source = df_source_locations[colnames['z']].to_list() # z coordinates 
        self.N_source = len(self.source_names) # number of potential sources

        
        
        # initialize the final simulated concentration array
        self.ch4_sim = np.zeros((self.N_t_sim, self.N_sensor)) # simulation in sim_dt resolution
        self.ch4_sim_res =  np.zeros((self.N_t_obs, self.N_sensor)) # simulation resampled to obs_dt resolution
        
    def _resample_inputs(self, dt_resample, resample_mode = 'interp'):
        '''
        Resample time_stamps, source_names, emision_rates, wind_speeds, wind_directions
        to target resolution.
        Inputs:
            dt_resample [s] (float): 
                the target time resolution to resample
            resample_mode (str): 
                method of resampling, 'pad' for padding, 'interp' for linear interpolation. 
        '''
        if resample_mode == 'pad': # pad for all quantities
            df = pd.DataFrame(data = {'source_names' : self.source_names,
                                      'emission_rates' : self.emission_rates, 
                                      'wind_speeds' : self.wind_speeds, 
                                      'wind_directions' : self.wind_directions}, 
                              index = self.time_stamps)
            df = df.resample(str(dt_resample)+'S').fillna(method="ffill")
            # resampled quantities:
            self.time_stamps_res = df.index.to_list() 
            self.emission_rates_res = df['emission_rates'].to_list() 
            self.source_names_res = df['source_names'].to_list() 
            self.wind_speeds_res = df['wind_speeds'].to_list() 
            self.wind_directions_res = df['wind_directions'].to_list() 
            
        elif resample_mode == 'interp': 
            # pad for source_name and emission_rate
            df1 = pd.DataFrame(data = {'source_names' : self.source_names,
                                      'emission_rates' : self.emission_rates}, 
                               index = self.time_stamps)
            df1 = df1.resample(str(dt_resample)+'S').fillna(method="ffill")
            self.time_stamps_res = df1.index.to_list() 
            self.emission_rates_res = df1['emission_rates'].to_list() 
            self.source_names_res = df1['source_names'].to_list() 
            
            # interpolate for wind_speeds and wind_directions:
            ## 1. convert ws & wd to x and y component of wind (denoted by u, v, respectively)
            ## 2. interpolate u and v
            ## 3. bring resampled u and v back to resampled ws and wd
            wind_u, wind_v = self._wind_vector_convert(self.wind_speeds, 
                                                       self.wind_directions,
                                                       direction = 'ws_wd_2_u_v')
            
            df2 = pd.DataFrame(data = {'wind_u' : wind_u,
                                       'wind_v' : wind_v}, 
                               index = self.time_stamps)
            df2 = df2.resample(str(dt_resample)+'S').interpolate()
            wind_u = df2['wind_u'].to_list()
            wind_v = df2['wind_v'].to_list()
            wind_speeds_res, wind_directions_res = \
            self._wind_vector_convert(wind_u, wind_v,direction= 'u_v_2_ws_wd') 
            self.wind_speeds_res = wind_speeds_res # type = list
            self.wind_directions_res = wind_directions_res # type = list
            
        else:
            raise NotImplementedError(">>>>> resample mode")
            
            
        
    def _extract_source_location(self, source_name):
        '''
        Find the coordinates for a given source
        Inputs:
            source_name (str): 
                the name of the given source
        Outputs:
            x0, y0, z0 [m] (scalars): 
                coordinate of the source
        '''
        idx = self.source_names.index(source_name)
        x0 = self.x_source[idx]
        y0 = self.y_source[idx]
        z0 = self.z_source[idx]
        
        return x0, y0, z0
    
    
    def _wind_vector_convert(self, input_1, input_2, direction = 'ws_wd_2_u_v'):
        '''
        Convert between (ws, wd) and (u,v)
        Inputs:
            input_1: 
                - list of wind speed [m/s] if direction = 'ws_wd_2_u_v'
                - list of the x component (denoted by u) of a wind vector [m/s] if direction = 'u_v_2_ws_wd'
            input_2: 
                - list of wind direction [degree] if direction = 'ws_wd_2_u_v'
                - list of the y component (denoted by v) of a wind vector [m/s] if direction = 'u_v_2_ws_wd'
            direction: 
                - 'ws_wd_2_u_v': convert wind speed and wind direction to x,y components of a wind vector
                - 'u_v_2_ws_wd': covnert x,y components of a wind vector to wind speed and wind directin
                       
        Outputs:
            quantities corresponding to the conversion direction
        '''
        if direction == 'ws_wd_2_u_v':
            ws, wd = input_1, input_2
            thetas = [270-x for x in wd] # convert wind direction to the angle between the wind vector and positive x-axis
            thetas = np.radians(thetas)
            u = [np.cos(theta)*l for l,theta in zip(ws, thetas)]
            v = [np.sin(theta)*l for l,theta in zip(ws, thetas)]
            output_1, output_2 = u, v
        elif direction == 'u_v_2_ws_wd':
            u, v = input_1, input_2
            ws = [(x**2 + y**2)**(1/2) for x,y in zip(u,v)]
            thetas = [np.arctan2(y, x) for x,y in zip(u,v)] # the angles between the wind vector and positive x-axis
            thetas = [x * 180 / np.pi for x in thetas]
            wd = [270 - x for x in thetas] # convert back to cardinal definition
            for i, x in enumerate(wd):
                if x < 0:
                    wd[i] = wd[i] + 360 
                elif x >= 360:
                    wd[i] = wd[i] - 360 
                else:
                    pass
            output_1, output_2 = ws, wd
        else:
            raise NotImplementedError(">>>>> wind vector conversion direction")
        
        return output_1, output_2
    
    def _get_downwind_coordinates(self, xs_old, ys_old, zs_old, x_0, y_0, wd):
        '''
        Convert coordinates from cardinal frame (east-north coordinate) to new frame 
        whose origin will be (x_0, y_0) 
        and whose positive x-axis be the down wind direction
        Inputs:
            xs_old, ys_old, zs_old [m] (list of floats): 
                coordinates of all sensors in the cardinal coordinates 
            x_0, y_0 [m] (scalar, float): 
                coordinate of the origin in the cardinal coordinates
            wd [degree] (scalar, float):  
                wind direction 
                (wd = 0, 90, 180, 270 <=> wind blowing from N, E, S, W, respectively)
        Outputs: 
            xs_new, ys_new, zs_new [m] (list of floats): 
                coordinates in the new frame 
        '''

        # translation
        xs_trans = [x - x_0 for x in xs_old]
        ys_trans = [y - y_0 for y in ys_old]

        # rotation
        theta = 270 - wd 
        theta = theta + 360 if theta < 0 else theta
        theta = np.deg2rad(theta)
        R = np.array(((np.cos(theta), np.sin(theta)), 
                      (-np.sin(theta), np.cos(theta)))) # rotation matrix in x-y plane
        result = np.matmul(R, np.array((xs_trans, ys_trans)))
        xs_new, ys_new, zs_new = result[0,:], result[1,:], zs_old # keep the z coordinates the same

        return list(xs_new), list(ys_new), list(zs_new)
    
    def _update_puff_location(self, x, y, z, ws, wd, dt):
        '''
        Update the puff location based on its previous location and 
        current wind speed and wind direction
        Inputs:
            x,y,z [m] (scalar): 
                previous location 
            ws [m/s] (scalar): 
                current wind speed 
            wd [degree] (scalar): 
                current wind direction 
            dt [s] (scalar): 
                time interval 
        Outputs:
            x_new, y_new, z_new [m] (scalar): 
                new location 
        '''
        theta = 270 - wd # theta is the angle between positive x-axis and the wind vector
        theta = theta + 360 if theta < 0 else theta
        theta = np.deg2rad(theta)
        x_new = x + ws * np.cos(theta) * dt
        y_new = y + ws * np.sin(theta) * dt
        z_new = z # assume the puff does not move vertically
        return x_new, y_new, z_new

    def _is_day(self, hour, day_range=(7, 19)):
        '''
        Determine if it is day hour or not.
        Inputs:
            hour (scalar, int): 
                a number between 0 - 23
            day_range (tuple of int): 
                a range to define day hours
        Output:
            (boolean): True if it is a day hour, False otherwise 
        '''
        if hour >= day_range[0] and hour <= day_range[1]:
            return True
        else:
            return False

    def _stability_classifier(self, wind_speed, hour):
        '''
        Determine the stability class based on Pasquill table (a reduced version)
        Inputs:
            wind_speed [m/s] (scalar, float): 
                wind speed
            hour (scalar, int): 
                hour of a day, a number between 0 and 23
        Outputs:
            stability_class (str): 
                stability class, a str from A-F
        '''
        is_day = self._is_day(hour)
        if wind_speed < 2:
            stability_class = "A" if is_day else "E"
        elif wind_speed >= 2 and wind_speed < 5:
            stability_class = "B" if is_day else "E"
        else:
            stability_class = "D"
        return stability_class
    
    def _get_sigma_coefficients(self, stability_class, x):
        '''
        Calculate sigma_{y,z} which are used in the Gausian puff equation
        sigma_z = a*x^b, x in km,
        sigma_y = 465.11628*x*tan(THETA) where THETA = 0.017453293*(c-d*ln(x)) where x in km
        Inputs:
            stability_class (str): 
                stability class, a str in A-F
            x [m] (scalar, float): 
                downwind distance from the source to sensor
        Outputs:
            sigma_y, sigma_z [m] (scalar, float): 
                sigma_{y,z} values
        '''

        # need to convert x to km
        x = x/1000

        flag = 0

        if x <= 0:
            sigma_y, sigma_z = (np.nan, np.nan) # set nan values for upwind locations
        else:
            # determine a,b,c,d values
            if stability_class == "A":
                if x < .1:
                    a, b = (122.800, 0.94470)
                elif x >= .1 and x < .15:
                    a, b = (158.080, 1.05420)
                elif x >= .15 and x < .20:
                    a, b = (170.220, 1.09320)
                elif x >= .20 and x < .25:
                    a, b = (179.520, 1.12620)
                elif x >= .25 and x < .30:
                    a, b = (217.410, 1.26440)
                elif x >= .30 and x < .40:
                    (a, b) = (258.890, 1.40940)
                elif x >= .40 and x < .50:
                    (a, b) = (346.750, 1.72830)
                elif x >= .5 and x < 3.11:
                    a, b = (453.850, 2.11660)
                else:
                    flag = 1 # sigma_z = 5000m in this case
                c, d = (24.1670, 2.5334)

            elif stability_class == "B":
                if x < .2:
                    a, b = (90.673, 0.93198)
                elif x >= .2 and x < .4:
                    a, b = (98.483, 0.98332)
                else:
                    a, b = (109.300, 1.09710)
                c, d = (18.3330, 1.8096)

            elif stability_class == "C":
                a, b = (61.141, 0.91465) 
                c, d = (12.5000, 1.0857)

            elif stability_class == "D":
                if x < .3:
                    a, b = (34.459, 0.86974)
                elif x >= .3 and x < 1:
                    a, b = (32.093, 0.81066)
                elif x >= 1 and x < 3:
                    a, b = (32.093, 0.64403)
                elif x >= 3 and x < 10:
                    a, b = (33.504, 0.60486)
                elif x >= 10 and x < 30:
                    a, b = (36.650, 0.56589)
                else:
                    a, b = (44.053, 0.51179)
                c, d = (8.3330, 0.72382)

            elif stability_class == "E":
                if x < .1:
                    a, b = (24.260, 0.83660)
                elif x >= .1 and x < .3:
                    a, b = (23.331, 0.81956)
                elif x >= 0.3 and x < 1:
                    a, b = (21.628, 0.75660)
                elif x >= 1 and x < 2:
                    a, b = (21.628, 0.63077)
                elif x >= 2 and x < 4:
                    a, b = (22.534, 0.57154)
                elif x >=4 and x < 10:
                    a, b = (24.703, 0.50527)
                elif x >=10 and x < 20:
                    a, b = (26.970, 0.46173)
                elif x >= 20 and x < 40:
                    a, b = (35.420, 0.37615)
                else:
                    a, b = (47.618, 0.29592)
                c, d = (6.2500, 0.54287)

            elif stability_class == "F":
                if x < .2:
                    a, b = (15.209, 0.81558)
                elif x >= .2 and x < .7:
                    a, b = (14.457, 0.78407)
                elif x >= .7 and x < 1:
                    a, b = (13.953, 0.68465)
                elif x >= 1 and x < 2:
                    a, b = (13.953, 0.63227)
                elif x >= 2 and x < 3:
                    a, b = (14.823, 0.54503)
                elif x >= 3 and x < 7:
                    a, b = (16.187, 0.46490)
                elif x >= 7 and x < 15:
                    a, b = (17.836, 0.41507)
                elif x >= 15 and x < 30:
                    a, b = (22.651, 0.32681)
                elif x >= 30 and x < 60:
                    a, b = (27.074, 0.27436)
                else:
                    a, b = (34.219, 0.21716)
                c, d = (4.1667, 0.36191)

            else:
                raise ValueError(">>>>> stability class")

            # calculate sigma_xyz
            Theta = 0.017453293 * (c-d*np.log(x)) # in radius
            sigma_y = 465.11628 * x * np.tan(Theta) # in meters

            if flag == 0:
                sigma_z = a * x**b # in meters
                sigma_z = np.min((sigma_z, 5000))
            else:
                sigma_z = 5000

        return sigma_y, sigma_z
        
    def _Gaussian_puff_equation(self, 
                                              q, ws, z0,
                                              xs, ys, zs, 
                                              sigma_ys, sigma_zs,
                                              ts,
                                              dtype = 'float32'):
        '''
        Gaussian puff equation where wind is assumed to be constant
        Inputs:
            q [kg] (scalar, float): 
                total emission corresponding to this puff
            ws [m/s] (scalar, float): 
                constant wind speed of the puff
            z0 [m] (scalar, float): 
                height of the source
            xs [m] (list of float, len = N_s): 
                x coordinates of all sensors
            ys [m] (list of float, len = N_s): 
                y coordinates of all sensors
            zs [m] (list of float, len = N_s): 
                z coordinates of all sensors
            sigma_ys [m] (list of float, len = N_s): 
                sigma value in y directin for all sensors
            sigma_zs [m] (list of float, len = N_s): 
                sigma value in z directin for all sensors
            ts [s] (list of float, len = n_t): 
                time stamp array for a single puff
            dtype (one of the TensorFlow data type): 
                data type of the tensors
        Outputs:
            c [ppm] (2D np.array, shape = [n_t, N_sensor]): 
                concentration matrix 
        '''
        # compuations are implemented in TensorFlow-GPU
        with tf.device("GPU:0"):
            # convert input variables to tensors
            xs = tf.constant(xs, shape = (1, self.N_sensor), dtype=dtype)
            ys = tf.constant(ys, shape = (1, self.N_sensor), dtype=dtype)
            zs = tf.constant(zs, shape = (1, self.N_sensor), dtype=dtype)
            sigma_ys = tf.constant(sigma_ys, shape = (1, self.N_sensor), dtype=dtype)
            sigma_zs = tf.constant(sigma_zs, shape = (1, self.N_sensor), dtype=dtype)
            ts = tf.constant(ts, shape = (len(ts), 1), dtype=dtype)
            # Gaussian puff equation:
            term_1 = q / ((2*np.pi)**(3/2) * (sigma_ys**2) * sigma_zs)
            term_2 = tf.math.exp(-(xs - ws*ts)**2 / (2*sigma_ys**2))
            term_3 = tf.math.exp(-ys**2 / (2*sigma_ys**2))
            term_4 = tf.math.exp(-(zs-z0)**2 / (2*sigma_zs**2)) + \
                     tf.math.exp(-(zs+z0)**2 / (2*sigma_zs**2))
            c = term_1 * term_2 * term_3 * term_4 * self.conversion_factor # concentration tensor, convert kg/s to ppm
            c = c.numpy()
            # convert all np.nan values to 0. This is becuase we set sigma_y,z = np.nan for upwind sensors
            c = np.nan_to_num(c, nan=0.0) 
            
        return c
    
                    
    def _concentration_per_puff(self, t_0):
        '''
        Compute the concentration time series for a single puff created at t_0
        Inputs:
            t_0 (scalar, pd.DateTime value): 
                starting time of the puff
        Outpus: 
            idx_0 (scalar, int): 
                index of the start of the time window in the entire time stamp frame
            idx_end (scalar, int): 
                index of the end of the time window in the entire time stamp frame
            c [ppm]: (2-D np.array, shape = [N_t_puff, N_sensor]): 
                simulated concentration       
        '''
        
        # extract sub data in the given time window
        t_end = t_0 + pd.Timedelta(self.puff_duration, 'S')
        t_end = min(t_end, self.time_stamps_res[-1]) # make sure the end time does not overflow
        idx_0, idx_end = self.time_stamps_res.index(t_0), self.time_stamps_res.index(t_end)
        n_t = idx_end - idx_0
        times = list(np.arange(0, n_t) * self.sim_dt) # time value array starting from 0
        source_name = self.source_names_res[idx_0]
        total_emission = self.emission_rates_res[idx_0] * self.puff_dt # quantity of emission for this puff
        wind_speed = self.wind_speeds_res[idx_0]
        wind_direction = self.wind_directions_res[idx_0]
        
        
        if source_name == self.non_emission_char:
            c = np.zeros((n_t, self.N_sensor)) # No emission ocurrs and hence no concentration
        else:
            x0, y0, z0 = self._extract_source_location(source_name) # source coordinates            
            # calculate sigma values
            ## determine stability class
            stability_class = self._stability_classifier(wind_speed, t_0.hour)
            ## convert receptor locations to downwind frame
            xs, ys, zs = self._get_downwind_coordinates(self.x_sensor, self.y_sensor, self.z_sensor, 
                                                        x0, y0, 
                                                        wind_direction) 
            ## initialize sigma value containers
            sigma_ys, sigma_zs = [], [] # with constant wind, sigma values don't change with time
            for x in xs:
                sigma_y, sigma_z = self._get_sigma_coefficients(stability_class, x)
                sigma_ys.append(sigma_y)
                sigma_zs.append(sigma_z)
            
            # compute concentrations
            c = self._Gaussian_puff_equation(total_emission,
                                                           wind_speed,
                                                           z0,
                                                           xs, ys, zs,
                                                           sigma_ys, sigma_zs,
                                                           times)
                                                               
        return idx_0, idx_end, c

    def _model_info_print(self):
        '''
        Print the parameters used in this model
        '''
        
        print("\n************************************************************")
        print("****************     PUFF SIMULATION START     *************")
        print("************************************************************")
        print(">>>>> start time: {}".format(datetime.datetime.now()))
        print(">>>>> configuration;")
        print("         Observation time resolution: {}[s]".format(self.obs_dt))
        print("         Simulation time resolution: {}[s]".format(self.sim_dt))
        print("         Puff creation time resolution: {}[s]".format(self.puff_dt))
        
        
        
        
        
    def simulation(self):
        '''
        Main code for simulation
        Outputs:
            ch4_sim_res [ppm] (2-D np.array, shape = [N_t_obs, N_sensor]): 
                simulated concentrations resampled according to observation dt
        '''
        if self.quiet == False:
            self._model_info_print()

        n_puff = len(self.time_stamps_puff_creation)
        
        # loop for each puff
        for i, t in enumerate(self.time_stamps_puff_creation): 
            # run simulation for the current puff
            idx_0, idx_end, c = self._concentration_per_puff(t)
            # add result from this puff to the entire concentration matrix
            self.ch4_sim[idx_0 : idx_end, :] += c 
            # report progress
            if self.quiet == False:
                if self.model_id == None:
                    if i % (n_puff // 10) == 0:
                        print('{}/10 done.'.format(i // (n_puff // 10)))
                else:
                    if i % (n_puff // 10) == 0:
                        print('Model {}: {}/10 done.'.format(self.model_id, i // (n_puff // 10)))
        
        # resample results to the obs_dt-resolution
        self.ch4_sim_res = self._resample_simulation(self.ch4_sim, self.obs_dt)
        
        
        if self.quiet == False:
            print("\n************************************************************")
            print("*****************    PUFF SIMULATION END     ***************")
            print("************************************************************")
            print(">>>>> end time:", datetime.datetime.now())
        else:
            print('*****************    PUFF SIMULATION END     ***************')
        
        return self.ch4_sim_res
    
    def _resample_simulation(self, c_matrix, dt, mode = 'mean'):
        '''
        Resample the simulation results 
        Inputs:
            c_matrix [ppm] (2D np.array, shape = [N_t_sim, N_s]): 
                the simulation results in sim_dt-resoltuion
                column order equals to the order of sensor_names 
            dt [s] (scalar, float): 
                the target time resolution
            mode (str):
                - 'mean': resmple by taking average 
                - 'resample': resample by taking every dt sample
        Outputs:
            c_matrix_res [ppm] (2D np.array, shape = [N_t_new, N_s]): 
                resampled simulation results 
        '''
        df = pd.DataFrame(c_matrix, 
                        # has to be commented or the end-to-end test breaks
                        # due to an increased number of columns
                        #   columns = self.sensor_names
                          index = self.time_stamps_res)
        if mode == 'mean':
            df = df.resample(str(dt)+'S').mean()
        elif mode == 'resample':
            df = df.resample(str(dt)+'S').asfreq()
        else:
            raise NotImplementedError(">>>>> sim to obs resampling mode") 
        
        c_matrix_res = df.to_numpy()
        
        return c_matrix_res
    
    
    
        
        
    