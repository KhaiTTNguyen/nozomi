# kurtosis/plot_by_diameter.py
from ..core.plotting import plot_metric_vs_parameter
from ..core.data_loader import load_experiment_data

def main():
    data = load_experiment_data('path/to/results')
    fig, ax = plot_metric_vs_parameter(
        data,
        metric='kurtosis',  # Only difference!
        parameter='diameter',
        compartment='intra',
        watson_k=20
    )