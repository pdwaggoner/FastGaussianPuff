# FastGaussianPuff

### ABOUT INPUT PARAMETERS:

There are three timestep related paremeters, sim_dt, obs_dt, and puff_dt. 
1. sim_dt: the time step size the model is simulated at in seconds. **Must be a positive integer.** This restriction is because of how the timestamps are handled in the C code. If this becomes an issue, we can likely find a workaround. 
2. obs_dt: this is the resolution we have observed wind data at in seconds. E.g. if we have 1-minute wind data, obs_dt = 60. **This paramater needs to be a positive integer multiple of sim_dt.**
3. puff_dt: this is how frequently new puffs are created in seconds, e.g. puff_dt = 1 means a new puff is created every second. The puff must be simulated until the puff moves off the grid, so creating fewer puffs will improve runtime at the tradeoff of accuracy. **Must be a positive integer.**

In the Python code, the wind data will be resampled from obs_dt to sim_dt by interpolation. So, if we only have 1-minute wind data, it will get interpolated down to the second for use in simulation. The concentration is computed at sim_dt resolution, but is resampled back up to obs_dt resolution at the end of the simulation. Both resolutions are accessible in ch4_sim and ch4_obs respectively. 

### Running on Casper
This code requires the Eigen C++ library, as well as the pybind11 and pandas python modules. Below is a guide to setting up the right environment on Casper.

1. Clone this repository on Casper
2. Activate the Eigen and Conda modules. Conda is used for managing the python packages. These modules can be activated by using the module system on Casper with the following commands:
```
$ module load eigen/3.4.0
$ module load conda
```
3. Create the conda environment

Once conda is activated, we need to create the conda environment. **This creation only needs to be done once**. The command below will create a new conda environment called `gp` from the provided .yml file.
```
conda env create -f conda_env.yml
```
This will install packages necessary to use the fastGaussianPuff code. To activate the environment, use
```
conda activate gp
```
If you need any other python packages for your code, add them to this environment using `conda install [package name]`. 

This will allow you to run the code in the current environment, however code should not be run on the login nodes. Either start an interactive session using `execcasper` or create a script to set up the environment and send jobs to the HPC queue. I'll try and create an example script of this type soon.

### Other notes

- Due to how the code is linked between Python and C++, you are unable to use Ctrl+C in a terminal to kill the simulation partway through. If you need to stop a simulation, kill the python process in the task manager or put the program in the background and kill it directly. On Linux, the shortcut to do this is Ctrl+Z followed by the command `kill %%`.

- To run the tests, you'll need the input data. It is currently stored on dropbox under fastGaussianPuff/test_data. Download that locally and copy it to Casper using scp.
