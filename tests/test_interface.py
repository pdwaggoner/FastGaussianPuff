import pytest
from FastGaussianPuff import GaussianPuff as GP
import pandas as pd
import numpy as np

start = pd.to_datetime('2022-01-05 04:04:00-06:00', utc=True)
end = pd.to_datetime('2022-01-05 04:34:00-06:00', utc=True)
Nt = 31
obs_dt = 60.0
sim_dt = 1.0
puff_dt = 1.0
source_coordinates = [[0, 0, 0]]
emission_rate = [1.0]
wind_speeds = np.repeat(1.0, Nt)
wind_directions = np.repeat(0.0, Nt)
grid_coords = [-50, -50, 0, 50, 50, 20]


@pytest.mark.parametrize("nx, ny, nz", [(20, 32, 19), (20, 32, 20)])
def test_good_grid_init(nx, ny, nz):
    params = {"obs_dt": obs_dt, "sim_dt": sim_dt, "puff_dt": puff_dt,
              "simulation_start": start, "simulation_end": end,
            "source_coordinates": source_coordinates, "emission_rates": emission_rate,
            "wind_speeds": wind_speeds, "wind_directions": wind_directions,
            "grid_coordinates": grid_coords}
    
    good_puff = GP(**params, nx=nx, ny=ny, nz=nz)

@pytest.mark.parametrize("nx, ny, nz", [(0, 32, 19), (20, -5, 20),(20, 32, 1)])
def test_bad_grid_init(nx, ny, nz):
    params = {"obs_dt": obs_dt, "sim_dt": sim_dt, "puff_dt": puff_dt,
              "simulation_start": start, "simulation_end": end,
            "source_coordinates": source_coordinates, "emission_rates": emission_rate,
            "wind_speeds": wind_speeds, "wind_directions": wind_directions,
            "grid_coordinates": grid_coords}
    
    with pytest.raises(ValueError):
        bad_puff = GP(**params, nx=nx, ny=ny, nz=nz)