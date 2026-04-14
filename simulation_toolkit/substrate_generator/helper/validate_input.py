import sys

def validate_parameters(params):
    """
    Validate input parameters to ensure they are within acceptable ranges.
    Exits the program with error messages if any parameter is out of range.
    """
    validation_rules = {
        'num_fibers': {
            'value': params.get('num_fibers'),
            'min': 50,
            'max': 1000,
            'description': 'Number of fibers'
        },
        'mean_diameter': {
            'value': params.get('mean_diameter'),
            'min': 0.5,
            'max': 9.0,
            'description': 'Mean diameter (µm)'
        },
        'orientation_shape_parameter': {
            'value': params.get('orientation_shape_parameter'),
            'min': 10,
            'max': 200,
            'description': 'Orientation parameter'
        },
        'bead_alpha_mean': {
            'value': params.get('bead_alpha_mean'),
            'min': 0.0,
            'max': 2.0,
            'description': 'Bead alpha mean (dimensionless)'
        },
        'bead_alpha_stdv': {
            'value': params.get('bead_alpha_stdv'),
            'min': 0.0,
            'max': 1.0,
            'description': 'Bead alpha stdv (dimensionless)'
        },
        'target_volume_fraction': {
            'value': params.get('target_volume_fraction'),
            'min': 0.01,
            'max': 0.98,
            'description': 'Volume fraction'
        }
    }
    
    errors = []
    
    for param_name, rules in validation_rules.items():
        value = rules['value']
        min_val = rules['min']
        max_val = rules['max']
        description = rules['description']
        
        # Check if parameter exists
        if value is None:
            errors.append(f"ERROR: Parameter '{param_name}' is missing from configuration")
            continue
            
        # Check if parameter is within valid range
        if not (min_val <= value <= max_val):
            errors.append(
                f"ERROR: {description} ({param_name}) = {value} is out of range.\n"
                f"       Acceptable range: {min_val} - {max_val}"
            )

    # Apply diameter-dependent volume fraction rule.
    mean_diameter = params.get('mean_diameter')
    target_volume_fraction = params.get('target_volume_fraction')
    if mean_diameter is not None and target_volume_fraction is not None:
        if 0.5 <= mean_diameter <= 1.0:
            if not (0.01 <= target_volume_fraction <= 0.98):
                errors.append(
                    "ERROR: Volume fraction (target_volume_fraction) is out of range for "
                    f"mean diameter {mean_diameter} um.\n"
                    "       Acceptable range: 0.01 - 0.98 when mean_diameter is 0.5 - 1.0 um"
                )
        else:
            if not (0.01 <= target_volume_fraction <= 0.7):
                errors.append(
                    "ERROR: Volume fraction (target_volume_fraction) is out of range for "
                    f"mean diameter {mean_diameter} um.\n"
                    "       Acceptable range: 0.01 - 0.7 when mean_diameter is above 1.0 um"
                )
    
    if errors:
        print("\n" + "="*60)
        print("INVALID PARAMETERS")
        print("="*60)
        for error in errors:
            print(error)
        
        print("\nValid parameter ranges:")
        print("-" * 40)
        for param_name, rules in validation_rules.items():
            print(f"  {rules['description']:25}: {rules['min']} - {rules['max']}")
        print("  Volume fraction (special): 0.01 - 0.98 for mean_diameter 0.5 - 1.0 um")
        print("                             0.01 - 0.7 for mean_diameter above 1.0 um")
        
        print("\nUpdate your substrate JSON config file with valid parameter choices.")
        print("="*60)
        sys.exit(1)
        
    print("All parameters validated for substrate generation!")