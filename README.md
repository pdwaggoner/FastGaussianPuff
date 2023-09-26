# FastGaussianPuff

ABOUT INPUT PARAMETERS:

There are three timestep related paremeters, sim_dt, obs_dt, and puff_dt. 
1. sim_dt: the time step size the model is simulated at in seconds. **Must be a positive integer.** This restriction is because of how the timestamps are handled in the C code. If this becomes an issue, we can likely find a workaround. 
2. obs_dt: this is the resolution we have observed wind data at in seconds. E.g. if we have 1-minute wind data, obs_dt = 60. **This paramater needs to be a positive integer multiple of sim_dt.**
3. puff_dt: this is how frequently new puffs are created in seconds, e.g. puff_dt = 1 means a new puff is created every second. The puff must be simulated until the puff moves off the grid, so creating fewer puffs will improve runtime at the tradeoff of accuracy. **Must be a positive integer.**

In the Python code, the wind data will be resampled from obs_dt to sim_dt by interpolation. So, if we only have 1-minute wind data, it will get interpolated down to the second for use in simulation. The concentration is computed at sim_dt resolution, but is resampled back up to obs_dt resolution at the end of the simulation. Both resolutions are accessible in ch4_sim and ch4_obs respectively. 

Due to how the code is linked between Python and C++, you are unable to use Ctrl+C in a terminal to kill the simulation partway through. If you need to stop a simulation, kill the python process in the task manager or put the program in the background and kill it directly. On Linux, the shortcut to do this is Ctrl+Z followed by the command `kill %%`.
