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
            'max': 0.7,
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
        
        print("\nUpdate your substrate JSON config file with valid parameter choices.")
        print("="*60)
        sys.exit(1)
        
    print("All parameters validated for substrate generation!")