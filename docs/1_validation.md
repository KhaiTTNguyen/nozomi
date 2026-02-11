## Validation Experiments
These experiments validate the reliability of the Monte Carlo diffusion engine via the following simulations:
* Free diffision in empty arena
* Intra and extra axonal compartmentation
* Periodic boundary condition handling 

#### How to run:
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/validation/narrow_pulse/test_03_periodic_boundary_xyz.py
```

#### Expected output: 
All 7 tests passed. 

Figures of spin positions pre- and post-simulations will be saved in 
```bash
/nozomi/tests/validation/narrow_pulse/test_figures/
```