# waveform_io.py
"""
Waveform I/O Module for Gradient Waveform Management

This module provides comprehensive functionality for saving and loading gradient
waveforms to/from text files. It supports both raw numpy arrays and complete
waveform objects with metadata preservation.

Classes supported:
- PGDiffWaveform
- CosineOGDiffWaveform
- Any numpy array-based waveforms

File formats:
- CSV format for raw arrays
- JSON+CSV hybrid format for complete objects

"""
import pickle
import json
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, Union, List
import csv
from datetime import datetime
import warnings

def save_waveform_object(waveform_obj,file_name):
    """
    Save a complete waveform object with all parameters and computed waveforms.
    
    Parameters:
    -----------
    waveform_obj : PGDiffWaveform, CosineOGDiffWaveform, or similar
        The waveform object to save
    file_name : str
        Path to the output file
    overwrite : bool, default False
        Whether to overwrite existing files
        
    Example:
    --------
    from gradient_waveforms import PGDiffWaveform
    waveform = PGDiffWaveform(duration=0.01, max_amplitude=1.0, time_step=1e-5)
    save_waveform_object(waveform, 'pg_waveform.json')
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name) if os.path.dirname(file_name) else '.', exist_ok=True)
    
    # Save pkl for waveform data
    pkl_file_name = os.path.join(file_name+'.pkl')
    with open(pkl_file_name, 'wb') as f:
        pickle.dump(waveform_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save JSON for waveform specs
    obj_data = {
        'timestamp': datetime.now().isoformat(),
        'class_name': waveform_obj.__class__.__name__
    }
    
    # Extract parameters (attributes that don't start with _)
    parameters = {}
    
    for attr_name in dir(waveform_obj):
        if not attr_name.startswith('_') and not callable(getattr(waveform_obj, attr_name)):
            attr_value = getattr(waveform_obj, attr_name)
            
            if isinstance(attr_value, np.ndarray):
                # Store array info for separate CSV files
                pass
            elif isinstance(attr_value, (int, float, str, bool, list, dict)) or attr_value is None:
                parameters[attr_name] = attr_value
            else:
                # Try to convert to string for other types
                parameters[attr_name] = str(attr_value)
    
    obj_data['parameters'] = parameters
    
    # Save JSON file
    with open(file_name+'.json', 'w') as f:
        json.dump(obj_data, f, indent=2)
    
    print('Done saving gradient data!')
          
def load_waveform_object(file_name):
    """
    Load and reconstruct a waveform object from saved files.
    
    Parameters:
    -----------
    file_name : str
        Path to the main pkl file
        
    Returns:
    --------
    waveform_obj : object
        Reconstructed waveform object (if class is available)
        If class cannot be imported, returns None
    obj_data : dict
        Complete object data including parameters and arrays
        
    Example:
    --------
     waveform, data = load_waveform_object('pg_waveform.json')
     if waveform is not None:
         print(f"Loaded {waveform.__class__.__name__}")
     else:
         print("Object data:", data['parameters'])
    """
    # Load main pkl file
    with open(file_name, 'rb') as file:
        # Unpickle the data
        loaded_data = pickle.load(file)
    return loaded_data
        
def load_file(file_name):
    # Direct approach using numpy.loadtxt
    data_array = np.loadtxt(file_name)
    return data_array