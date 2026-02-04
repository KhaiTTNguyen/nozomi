# adc/plot_by_diameter.py
from ..core import plot_metric_vs_parameter
from ..core import load_experiment_data

def main():
    data = load_experiment_data('path/to/results')
    fig, ax = plot_metric_vs_parameter(
        data, 
        metric='adc',
        parameter='diameter',
        compartment='intra',
        watson_k=20
    )
    # Save with appropriate naming