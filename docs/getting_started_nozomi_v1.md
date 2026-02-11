# NOZOMI Substrate Generation - Getting Started

## Quick Start (5 minutes)

### 1. Copy and edit a template
```bash
cd nozomi
cp experiment/setup/substrate/template-substrate.json my-first-substrate.json
```

Edit `my-first-substrate.json` - remove all lines starting with `_` and adjust:
```json
{
  "experiment_name": "my_first_substrate",
  "parameters": {
      "orientation_shape_parameter": 20,
      "target_volume_fraction": 0.5,
      "num_fibers": 50,
      "mean_diameter": 2.0,
      "sigma_diameter": 0.5,
      "spheres_spacing": 0.5,
      "repeats": 1
  }
}
```

### 2. Generate substrate
```bash
./bin/run-geometry-gen.sh --config=my-first-substrate.json --gpu=0
```

### 3. Check results
```bash
ls experiment/result/  # Find your output folder
```

## Common Parameter Presets

### Corpus Callosum (highly aligned)
```json
{
  "orientation_shape_parameter": 200,
  "target_volume_fraction": 0.7, 
  "num_fibers": 500,
  "mean_diameter": 1.2
}
```

### Cortical White Matter (moderate alignment)  
```json
{
  "orientation_shape_parameter": 50,
  "target_volume_fraction": 0.5,
  "num_fibers": 300,
  "mean_diameter": 2.0
}
```

### Crossing Region (dispersed)
```json
{
  "orientation_shape_parameter": 10,
  "target_volume_fraction": 0.4,
  "num_fibers": 200,
  "mean_diameter": 2.5
}
```

## Key Parameters to Adjust

| What you want | Parameter to change | Values |
|---------------|-------------------|---------|
| More aligned fibers | `orientation_shape_parameter` | 50-200 |
| More dispersed fibers | `orientation_shape_parameter` | 5-20 |
| Denser packing | `target_volume_fraction` | 0.6-0.7 |
| Sparse packing | `target_volume_fraction` | 0.3-0.4 |
| Thicker axons | `mean_diameter` | 3.0-4.0 |
| Thinner axons | `mean_diameter` | 1.0-2.0 |
| Higher resolution | `spheres_spacing` | 0.2-0.4 |
| Faster generation | `spheres_spacing` | 0.7-1.0 |

## Troubleshooting

**Generation takes too long?**
- Reduce `num_fibers` to 50-100
- Increase `spheres_spacing` to 0.7

**Generation fails?** 
- Lower `target_volume_fraction` to 0.4-0.5
- Reduce `num_fibers`

**GPU error?**
- Check: `nvidia-smi`
- Try different GPU: `--gpu=1`

## Next Steps

1. **Read the full documentation**: [substrate_generation.md](substrate_generation.md)
2. **Browse existing examples**: `experiment/setup/substrate/single_substrate/`
3. **Analyze your results**: Use the visualization plots in `experiment/result/`
4. **Run diffusion simulation**: Follow the Monte Carlo diffusion guide

## Need Help?

- Check parameter validation ranges in the full documentation
- Look at existing configuration examples for similar use cases  
- Examine the output visualizations to verify results
- Test with smaller `num_fibers` first to iterate quickly