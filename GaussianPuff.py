#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:59:36 2022

@author: mengjia, rykerfish
"""
import numpy as np
import pandas as pd
import datetime
from math import floor
import CGaussianPuff as C_GP
import CSensorGaussianPuff as CS_GP

class GaussianPuff:
    def __init__(self,
                 obs_dt, sim_dt, puff_dt, 
                 simulation_start, simulation_end,
                 source_coordinates, emission_rates,
                 wind_speeds, wind_directions,
                 grid_coordinates=None,
                 sensor_coordinates=None,
                 nx=None, ny=None, nz=None,
                 using_sensors=False,
                 puff_duration = None,
                 exp_threshold_tolerance = 1e-9,
                 conversion_factor = 1e6*1.524,
                 quiet=False):
        
        '''
        Inputs: 
            wind_speeds [m/s] (list of floats): 
                wind speed at each time stamp, in obs_dt resolution
            wind_directions [degree] (list of floats): 
                wind direction at each time stamp, in obs_dt resolution.
                follows the conventional definition: 
                0 -> wind blowing from North, 90 -> E, 180 -> S, 270 -> W
            obs_dt [s] (scalar, int): 
                time interval (dt) for the observations
                NOTE: obs_dt must be a positive integer multiple of sim_dt due to how resampling is done. 
            sim_dt [s] (scalar, int): 
                time interval (dt) for the simulation results
                NOTE: must be an integer number of seconds due to how timestamps are handled in C.
            puff_dt [s] (scalar, int): 
                time interval (dt) between two successive puffs' creation
            simulation_start, simulation_end (pd.DateTime values)
                start and end times for the emission to be simulated.
                NOTE: should be a minute resolution.
            source_coordinates (array, size=(n_sources, 3)) [m]:
                holds source coordinates in (x,y,z) format in meters for each source.
            emission_rates: (array, length=n_sources) [kg/hr]:
                rate that each source is emitting at in kg/hr.
            grid_coordinates: (array, length=6) [m]
                holds the coordinates for the corners of the grid to be created.
                format is grid_coordinates=[min_x, min_y, min_z, max_x, max_y, max_z]
            using_sensors (boolean):
                If True, ignores grid-related input parameters and only simulates at sensor locations
                given in df_sensor_locations.
            nx, ny, ny (scalar, int):
                Number of points for the grid the x, y, and z directions
            puff_duration [s] (int):
                how many seconds a puff can 'live'; we assume a puff will fade away after a certain time
            exp_threshold_tolerance (scalar, float):
                the tolerance used to threshold the exponentials when evaluating the Gaussian equation
            conversion_factor (scalar, float): 
                convert from kg/m^3 to ppm, this factor is for ch4 only
            quiet (boolean): 
                if output progress information while running or not
        '''

        self._verify_inputs(sim_dt, puff_dt, obs_dt)

        self.obs_dt = obs_dt 
        self.sim_dt = sim_dt 
        self.puff_dt = puff_dt


        self.sim_start = simulation_start
        self.sim_end = simulation_end

        self.quiet = quiet

        ns = (simulation_end-simulation_start).total_seconds() + 60 # +1 minute for (end-start) + 1
        self.n_obs = floor(ns/obs_dt) # number of observed data points we have

        # resample the wind data from obs_dt to the simulation resolution sim_dt
        self._interpolate_wind_data(wind_speeds, wind_directions, puff_dt, simulation_start, simulation_end)

        # save timeseries of simulation resolution so we can resample back to observation later
        self.time_stamps_sim = pd.date_range(self.sim_start, self.sim_end, freq=str(self.sim_dt)+"S")
        self.n_sim = len(self.time_stamps_sim) # number of simulation time steps

        # by default, don't have puff duration since the C code computes the correct time cutoff for each sim
        # if a puff default is set, you'll loose accuracy on puffs that remain on the grid for a long time
        # due to atmospheric conditions
        if puff_duration == None:
            puff_duration = self.n_sim # ensures we don't overflow time index

        # if using_sensors:
        #     self.using_sensors = True
        #     self.X = np.array(self.x_sensor)
        #     self.Y = np.array(self.y_sensor)
        #     self.Z = np.array(self.z_sensor)

        #     self.nx = self.N_sensor
        #     self.ny = 1
        #     self.nz = 1
        #     self.N_points = self.N_sensor
        #     self.grid_dims = (self.nx, self.ny, self.nz)

        # creates grid
        if not using_sensors:
            self.using_sensors = False

            self.nx = nx
            self.ny = ny
            self.nz = nz
            self.N_points = self.nx*self.ny*self.nz

            x_min = grid_coordinates[0]
            y_min = grid_coordinates[1]
            z_min = grid_coordinates[2]
            x_max = grid_coordinates[3]
            y_max = grid_coordinates[4]
            z_max = grid_coordinates[5]

            x, y, z = np.linspace(x_min, x_max, self.nx), np.linspace(y_min, y_max, self.ny), np.linspace(z_min, z_max, self.nz)
            
            self.X, self.Y, self.Z = np.meshgrid(x, y, z) # x-y-z grid across site in utm
            self.grid_dims = np.shape(self.X)

            # work with the flattened grids
            self.X = self.X.ravel()
            self.Y = self.Y.ravel()
            self.Z = self.Z.ravel()

            # constructor for the c code
            self.GPC = C_GP.CGaussianPuff(
                    self.X, self.Y, self.Z, 
                    self.nx, self.ny, self.nz, 
                    sim_dt, puff_dt, puff_duration,
                    simulation_start, simulation_end,
                    self.wind_speeds_sim, self.wind_directions_sim,
                    source_coordinates, emission_rates,
                    conversion_factor, exp_threshold_tolerance, quiet)
        else:
            self.using_sensors = True
            self.N_points = len(sensor_coordinates)

            self.X, self.Y, self.Z = [], [], []
            for sensor in sensor_coordinates:
                self.X.append(sensor[0])
                self.Y.append(sensor[1])
                self.Z.append(sensor[2])

            self.GPC = CS_GP.CSensorGaussianPuff(
                self.X, self.Y, self.Z, 
                self.N_points,
                sim_dt, puff_dt, puff_duration,
                simulation_start, simulation_end,
                self.wind_speeds_sim, self.wind_directions_sim,
                source_coordinates, emission_rates,
                conversion_factor, exp_threshold_tolerance, quiet
            )


        # initialize the final simulated concentration array
        self.ch4_sim = np.zeros((self.n_sim, self.N_points)) # simulation in sim_dt resolution, flattened
        # self.ch4_obs =  np.zeros((self.n_obs, self.N_points)) # simulation resampled to obs_dt resolution

    def _verify_inputs(self, sim_dt, puff_dt, obs_dt):

        if not isinstance(sim_dt, int) or sim_dt <= 0:
            print("ERROR IN INITIALIZATION: sim_dt must be a positive integer value")
            exit(-1)

        if not isinstance(puff_dt, int) or puff_dt <= 0:
            print("ERROR IN INITIALIZATION: puff_dt must be a positive integer value")
            exit(-1)

        if obs_dt % sim_dt != 0:
            print("ERROR IN INITIALIZATION: obs_dt must be a positive integer multiple of sim_dt")
            exit(-1)

        if puff_dt % sim_dt != 0:
            print("ERROR IN INITIALIZATION: sim_dt must be a positive integer multiple of puff_dt")



    def _interpolate_wind_data(self, wind_speeds, wind_directions, puff_dt, sim_start, sim_end):
        '''
        Resample wind_speeds and wind_directions to the simulation resolution by interpolation.
        Inputs:
            sim_dt [s] (int): 
                the target time resolution to resample to
            sim_start, sim_end (pd.DateTime)
                DateTimes the simulation start and ends at
            n_obs (int)
                number of observation data points across the simulation time range
        '''

        # creates a timeseries at obs_dt resolution
        time_stamps = pd.date_range(sim_start, sim_end, self.n_obs)

        # interpolate for wind_speeds and wind_directions:
        ## 1. convert ws & wd to x and y component of wind (denoted by u, v, respectively)
        ## 2. interpolate u and v
        ## 3. bring resampled u and v back to resampled ws and wd
        wind_u, wind_v = self._wind_vector_convert(wind_speeds, 
                                                    wind_directions,
                                                    direction = 'ws_wd_2_u_v')


        # resamples wind data to sim_dt resolution
        wind_df = pd.DataFrame(data = {'wind_u' : wind_u,
                                    'wind_v' : wind_v}, 
                            index = time_stamps)
        
        wind_df = wind_df.resample(str(puff_dt)+'S').interpolate()
        wind_u = wind_df['wind_u'].to_list()
        wind_v = wind_df['wind_v'].to_list()

        self.wind_speeds_sim, self.wind_directions_sim = self._wind_vector_convert(wind_u, wind_v,direction= 'u_v_2_ws_wd') 

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
        
    def simulate(self):
        '''
        Main code for simulation
        Outputs:
            ch4_sim_res [ppm] (2-D np.array, shape = [N_t_obs, N_sensor]): 
                simulated concentrations resampled according to observation dt
        '''
        if self.quiet == False:
            self._model_info_print()

        self.GPC.simulate(self.ch4_sim)

        # resample results to the obs_dt-resolution
        self.ch4_obs = self._resample_simulation(self.ch4_sim, self.obs_dt)
        
        
        if self.quiet == False:
            print("\n************************************************************")
            print("*****************    PUFF SIMULATION END     ***************")
            print("************************************************************")
        
        return self.ch4_obs
    
    def _resample_simulation(self, c_matrix, obs_dt, mode = 'mean'):
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

        df = pd.DataFrame(c_matrix, index = self.time_stamps_sim)
        if mode == 'mean':
            df = df.resample(str(obs_dt)+'S').mean()
        elif mode == 'resample':
            df = df.resample(str(obs_dt)+'S').asfreq()
        else:
            raise NotImplementedError(">>>>> sim to obs resampling mode") 
        
        c_matrix_res = df.to_numpy()
        # c_matrix_res = np.reshape(c_matrix_res, (self.n_obs, *self.grid_dims)) # reshape to grid coordinates
        
        return c_matrix_res
