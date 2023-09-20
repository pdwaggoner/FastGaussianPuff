#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:59:36 2022

@author: mengjia
"""
import numpy as np
import pandas as pd
import datetime
# import numba as nb
import time
import CGaussianPuff as C_GP

class vectorizedGAUSSIANPUFF:
    def __init__(self, 
                 time_stamps, source_names, emission_rates, wind_speeds, wind_directions,
                 obs_dt, sim_dt, puff_dt, 
                 df_sensor_locations, df_source_locations,
                 using_sensors=True,
                 x_num=None, y_num=None, z_num=None,
                 two_dimensional_grid=False,
                 z_height=None,
                 puff_duration = 1080,
                 exp_threshold_tolerance = 1e-9,
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
            using_sensors (boolean):
                If True, ignores grid-related input parameters and only simulates at sensor locations
                given in df_sensor_locations.
            two_dimensional_grid (boolean):
                If True, ignores z_num as an input and only creates a 2D grid in XY. If z_height is set,
                the 2D grid is placed at that height.
            z_height [m] (scalar, float):
                Height at which the 2D grid is created at. Only used if two_dimensional_grid=True (above)
            x_num, y_num, z_num (scalar, int):
                Number of points for the grid the x, y, and z directions
            puff_duration [s] (int):
                how many seconds a puff can 'live'; we assume a puff will fade away after a certain time
            exp_threshold_tolerance (scalar, float):
                the tolerance used to threshold the exponentials when evaluating the Gaussian equation
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
        self.exp_thresh_tol = exp_threshold_tolerance
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

        if using_sensors:
            self.using_sensors = True
            self.X = np.array(self.x_sensor)
            self.Y = np.array(self.y_sensor)
            self.Z = np.array(self.z_sensor)

            self.nx = self.N_sensor
            self.ny = 1
            self.nz = 1
            self.N_points = self.N_sensor
            self.grid_dims = (self.nx, self.ny, self.nz)

        if not using_sensors:
            self.using_sensors = False

            # parameters for grid
            self.nx = x_num
            self.ny = y_num
            if two_dimensional_grid:
                self.nz = 1
            else:
                self.nz = z_num
            self.N_points = self.nx*self.ny*self.nz

            # set up the grid
            x_min = np.mean(self.x_sensor) - 2*np.std(self.x_sensor)
            x_max = np.mean(self.x_sensor) + 2*np.std(self.x_sensor)
            y_min = np.mean(self.y_sensor) - 2*np.std(self.y_sensor)
            y_max = np.mean(self.y_sensor) + 2*np.std(self.y_sensor)

            if two_dimensional_grid: # setup for 2D grid
                if z_height is None:
                    print("Warning: No height for 2D grid specified. Using METEC default of 2.4. \
                          To specify a height, set z_height as a parameter during initialization.")
                    z_min = 2.4
                    z_max = 2.4
                else:
                    z_min = z_height
                    z_max = z_height
            else: # standard setup for 3D grid
                z_min = 0
                z_max = 10*max(self.z_sensor) # might need to make this higher (max of source and sensor?)

            x, y, z = np.linspace(x_min, x_max, self.nx), np.linspace(y_min, y_max, self.ny), np.linspace(z_min, z_max, self.nz)
            
            self.X, self.Y, self.Z = np.meshgrid(x, y, z) # x-y-z grid across site in utm
            self.grid_dims = np.shape(self.X)

            # work with the flattened grids
            self.X = self.X.ravel()
            self.Y = self.Y.ravel()
            self.Z = self.Z.ravel()
        
        # map_table = np.zeros(self.grid_dims, dtype=np.int32).tolist()

        # constructor for the c code
        self.GPC = C_GP.CGaussianPuff(self.X, self.Y, self.Z, 
                                      self.nx, self.ny, self.nz, 
                                      self.sim_dt,
                                      self.conversion_factor, self.exp_thresh_tol)

        # initialize the final simulated concentration array
        self.ch4_sim = np.zeros((self.N_t_sim, self.N_points)) # simulation in sim_dt resolution, flattened
        self.ch4_sim_res =  np.zeros((self.N_t_obs, *self.grid_dims)) # simulation resampled to obs_dt resolution

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
            df = df.resample(str(dt_resample)+'S').ffill()
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
            df1 = df1.resample(str(dt_resample)+'S').ffill()
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
    
    def _get_downwind_coordinates(self, X, Y, x_0, y_0, wd):
        '''
        Convert coordinates from cardinal frame (east-north coordinate) to new frame 
        whose origin will be (x_0, y_0) 
        and whose positive x-axis be the down wind direction
        Inputs:
            X, Y, Z [m] (np.arrays, floats from np.meshgrid, size = (y.num, x.num, z.num)): 
                coordinates of grid points in the cardinal coordinates 
            x_0, y_0 [m] (scalar, float): 
                coordinate of the origin (source location) in the cardinal coordinates
            wd [degree] (scalar, float):  
                wind direction 
                (wd = 0, 90, 180, 270 <=> wind blowing from N, E, S, W, respectively)
        Outputs: 
            X_rot, Y_rot, Z [m] (np.arrays, floats from np.meshgrid): 
                coordinates of grid points in the new frame 
        '''

        # translation
        X_trans = X - x_0
        Y_trans = Y - y_0

        # rotation
        theta = 270 - wd 
        theta = theta + 360 if theta < 0 else theta
        theta = np.deg2rad(theta)

        R = np.array(((np.cos(theta), np.sin(theta)), 
                    (-np.sin(theta), np.cos(theta)))) # rotation matrix in x-y plane

        rotated = np.matmul(R, np.vstack([X_trans, Y_trans]))

        return rotated[0], rotated[1] # rotated versions of the flattened X and Y grids

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
  
    def _concentration_per_puff(self, t_0):
        '''
        Compute the concentration time series for a single puff created at t_0
        Inputs:
            t_0 (scalar, pd.DateTime value): 
                starting time of the puff
        Outputs: 
            None. Instead, the concentration time series for the single puff is added
            to the overall concentration time series ch4_sim inside of the gaussian_puff_equation call.
        '''
        # extract sub data in the given time window
        t_end = t_0 + pd.Timedelta(self.puff_duration, 'S')
        t_end = min(t_end, self.time_stamps_res[-1]) # make sure the end time does not overflow
        idx_0, idx_end = self.time_stamps_res.index(t_0), self.time_stamps_res.index(t_end)
        n_t = idx_end - idx_0
        source_name = self.source_names_res[idx_0]
        total_emission = self.emission_rates_res[idx_0] * self.puff_dt # quantity of emission for this puff
        wind_speed = self.wind_speeds_res[idx_0]
        wind_direction = self.wind_directions_res[idx_0]
        
        if source_name == 'None':
            return # No emission ocurrs and hence no concentration
        elif n_t == 0:
            return # No time steps so no concentration to compute (happens at end of time block)    
        else:

            x0, y0, z0 = self._extract_source_location(source_name) # get source coordinates   

            stability_class = self._stability_classifier(wind_speed, t_0.hour) # determine stability class

            self.GPC.concentrationPerPuff(total_emission, wind_direction, wind_speed,
                                x0, y0, z0,
                                stability_class,
                                self.ch4_sim[idx_0:idx_end])



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
        if self.using_sensors:
            print("         Running in sensor mode")
        else:
            print(f"         Running in grid mode with grid dimensions {self.grid_dims}")
        
        
        
        
        
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
            self._concentration_per_puff(t)

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
            c_matrix [ppm] (2D np.ndarray, shape = [N_t_sim, self.N_points]): 
                the simulation results in sim_dt resolution across the whole grid
            dt [s] (scalar, float): 
                the target time resolution
            mode (str):
                - 'mean': resmple by taking average 
                - 'resample': resample by taking every dt sample
        Outputs:
            c_matrix_res [ppm] (4D np.array, shape = [N_t_new, self.grid_dims)]): 
                resampled simulation results 
        '''

        # flattens only the grid but leaves the time series in order
        # c_flattened = c_matrix.reshape(self.N_t_sim, self.N_points)

        df = pd.DataFrame(c_matrix, index = self.time_stamps_res)
        if mode == 'mean':
            df = df.resample(str(dt)+'S').mean()
        elif mode == 'resample':
            df = df.resample(str(dt)+'S').asfreq()
        else:
            raise NotImplementedError(">>>>> sim to obs resampling mode") 
        
        c_matrix_res = df.to_numpy()
        c_matrix_res = np.reshape(c_matrix_res, (self.N_t_obs, *self.grid_dims)) # reshape to grid coordinates
        
        return c_matrix_res
    
    
    
        
        
    