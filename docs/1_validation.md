## Validation Experiments
These experiments validate the reliability of the Monte Carlo diffusion engine via the following simulations:
* Free diffision in empty arena
* Intra and extra axonal compartmentation
* Periodic boundary condition handling 

The experiements are set to run for `100000 spins`, a time step of `dt = 0.002 ms`, a total of `nt = 50000` time steps, leading to a diffusion time of `100 ms`.
#### How to run:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/validation/narrow_pulse/test_03_periodic_boundary_xyz.py
```

#### Expected output: 
All 7 tests passed. All tests should take ~11 minutes to run.

Figures of spin positions pre- and post-simulations will be saved in 
```bash
/nozomi/tests/validation/narrow_pulse/test_figures/
```