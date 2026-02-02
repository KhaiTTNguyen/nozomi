# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# def load_and_validate_data(csv_file_path):
#     """
#     Load CSV data and validate required columns.
    
#     Args:
#         csv_file_path (str): Path to the CSV file
        
#     Returns:
#         pd.DataFrame: Validated dataframe
#     """
#     try:
#         df = pd.read_csv(csv_file_path)
#         print(f"Successfully loaded data with shape: {df.shape}")
        
#         # Required columns (note: user's dataset may have 'mae'/'std_mae' instead of 'mse'/'std_mse')
#         required_base_cols = ['molecules', 'time_step', 'time_threshold', 
#                              'min_mse', 'max_mse', 'mean_computation_time', 
#                              'std_computation_time', 'n_successful_runs']
        
#         # Check for MSE columns first, then fall back to MAE columns
#         if 'mse' in df.columns and 'std_mse' in df.columns:
#             mse_cols = ['mse', 'std_mse']
#             print("Found MSE columns in dataset")
#         elif 'mae' in df.columns and 'std_mae' in df.columns:
#             # User's current dataset has MAE columns - we'll rename them
#             df = df.rename(columns={'mae': 'mse', 'std_mae': 'std_mse'})
#             mse_cols = ['mse', 'std_mse']
#             print("Found MAE columns - renamed to MSE for analysis")
#             print("Note: This script assumes MAE and MSE have similar analysis patterns")
#         else:
#             raise ValueError("Neither 'mse'/'std_mse' nor 'mae'/'std_mae' columns found")
        
#         required_cols = required_base_cols + mse_cols
        
#         # Check for missing columns
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             print(f"Warning: Missing columns: {missing_cols}")
#             # Try to continue with available columns
#             available_cols = [col for col in required_cols if col in df.columns]
#             print(f"Available columns: {available_cols}")
        
#         # Validate data types and handle missing values
#         numeric_cols = ['molecules', 'time_step', 'mse', 'std_mse', 'mean_computation_time']
#         for col in numeric_cols:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
        
#         # Remove rows with NaN values in critical columns
#         critical_cols = ['molecules', 'time_step', 'mse']
#         df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
#         print(f"Data validation complete. Final shape: {df.shape}")
#         print(f"Unique molecules: {sorted(df['molecules'].unique())}")
#         print(f"Unique time steps: {sorted(df['time_step'].unique())}")
        
#         return df
        
#     except FileNotFoundError:
#         raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
#     except Exception as e:
#         raise Exception(f"Error loading data: {str(e)}")

# def setup_plot_style():
#     """Set up consistent plot styling"""
#     plt.style.use('default')
#     sns.set_palette("husl")
#     plt.rcParams.update({
#         'font.size': 12,
#         'axes.labelsize': 14,
#         'axes.titlesize': 16,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 11,
#         'figure.titlesize': 18,
#         'lines.linewidth': 2,
#         'grid.alpha': 0.3
#     })

# def plot_mse_vs_molecules(df, output_dir='plots'):
#     """
#     Plot MSE vs number of molecules for different time steps (log-log scale)
#     """
#     plt.figure(figsize=(12, 8))
    
#     time_steps = sorted(df['time_step'].unique())
#     colors = plt.cm.tab10(np.linspace(0, 1, len(time_steps)))
    
#     for i, ts in enumerate(time_steps):
#         subset = df[df['time_step'] == ts].sort_values('molecules')
        
#         if len(subset) > 0:
#             plt.errorbar(subset['molecules'], subset['mse'], 
#                         yerr=subset['std_mse'] if 'std_mse' in subset.columns else None,
#                         marker='o', linestyle='-', linewidth=2, markersize=6,
#                         color=colors[i], label=f'Time Step {ts}',
#                         capsize=5, capthick=2, alpha=0.8)
    
#     plt.xlabel('Number of Molecules')
#     plt.ylabel('Mean Square Error (MSE)')
#     plt.title('MSE vs Number of Molecules (Different Time Steps)')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     # Save plot
#     Path(output_dir).mkdir(exist_ok=True)
#     plt.savefig(f'{output_dir}/mse_vs_molecules_loglog.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved: {output_dir}/mse_vs_molecules_loglog.png")

# def plot_mse_vs_timestep(df, output_dir='plots'):
#     """
#     Plot MSE vs time step for different numbers of molecules (log-log scale)
#     """
#     plt.figure(figsize=(12, 8))
    
#     molecules = sorted(df['molecules'].unique())
#     colors = plt.cm.viridis(np.linspace(0, 1, len(molecules)))
    
#     for i, mol in enumerate(molecules):
#         subset = df[df['molecules'] == mol].sort_values('time_step')
        
#         if len(subset) > 0:
#             plt.errorbar(subset['time_step'], subset['mse'],
#                         yerr=subset['std_mse'] if 'std_mse' in subset.columns else None,
#                         marker='s', linestyle='-', linewidth=2, markersize=6,
#                         color=colors[i], label=f'{mol} Molecules',
#                         capsize=5, capthick=2, alpha=0.8)
    
#     plt.xlabel('Time Step')
#     plt.ylabel('Mean Square Error (MSE)')
#     plt.title('MSE vs Time Step (Different Number of Molecules)')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     # Save plot
#     Path(output_dir).mkdir(exist_ok=True)
#     plt.savefig(f'{output_dir}/mse_vs_timestep_loglog.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved: {output_dir}/mse_vs_timestep_loglog.png")

# def plot_computation_time_vs_molecules(df, output_dir='plots'):
#     """
#     Plot computation time vs number of molecules
#     """
#     plt.figure(figsize=(12, 8))
    
#     time_steps = sorted(df['time_step'].unique())
#     colors = plt.cm.plasma(np.linspace(0, 1, len(time_steps)))
    
#     for i, ts in enumerate(time_steps):
#         subset = df[df['time_step'] == ts].sort_values('molecules')
        
#         if len(subset) > 0 and 'mean_computation_time' in subset.columns:
#             plt.errorbar(subset['molecules'], subset['mean_computation_time'],
#                         yerr=subset['std_computation_time'] if 'std_computation_time' in subset.columns else None,
#                         marker='^', linestyle='-', linewidth=2, markersize=6,
#                         color=colors[i], label=f'Time Step {ts}',
#                         capsize=5, capthick=2, alpha=0.8)
    
#     plt.xlabel('Number of Molecules')
#     plt.ylabel('Computation Time (seconds)')
#     plt.title('Computation Time vs Number of Molecules')
#     plt.xscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     # Save plot
#     Path(output_dir).mkdir(exist_ok=True)
#     plt.savefig(f'{output_dir}/computation_time_vs_molecules.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved: {output_dir}/computation_time_vs_molecules.png")

# def plot_mse_bar_grouped(df, output_dir='plots'):
#     """
#     Create a bar plot of MSE vs molecules grouped by time steps
#     """
#     plt.figure(figsize=(14, 8))
    
#     # Prepare data for grouped bar plot
#     molecules = sorted(df['molecules'].unique())
#     time_steps = sorted(df['time_step'].unique())
    
#     x = np.arange(len(molecules))
#     width = 0.8 / len(time_steps)
    
#     colors = plt.cm.Set3(np.linspace(0, 1, len(time_steps)))
    
#     for i, ts in enumerate(time_steps):
#         mse_values = []
#         error_values = []
        
#         for mol in molecules:
#             subset = df[(df['molecules'] == mol) & (df['time_step'] == ts)]
#             if len(subset) > 0:
#                 mse_values.append(subset['mse'].iloc[0])
#                 error_values.append(subset['std_mse'].iloc[0] if 'std_mse' in subset.columns else 0)
#             else:
#                 mse_values.append(0)
#                 error_values.append(0)
        
#         plt.bar(x + i * width, mse_values, width, 
#                yerr=error_values, capsize=3,
#                color=colors[i], alpha=0.8, 
#                label=f'Time Step {ts}')
    
#     plt.xlabel('Number of Molecules')
#     plt.ylabel('Mean Square Error (MSE)')
#     plt.title('MSE Comparison Across Molecules and Time Steps')
#     plt.xticks(x + width * (len(time_steps) - 1) / 2, molecules)
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3, axis='y')
#     plt.legend()
#     plt.tight_layout()
    
#     # Save plot
#     Path(output_dir).mkdir(exist_ok=True)
#     plt.savefig(f'{output_dir}/mse_bar_grouped.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved: {output_dir}/mse_bar_grouped.png")

# def plot_mse_std_vs_molecules(df, time_step_filter=None, output_dir='plots'):
#     """
#     Plot standard deviation of MSE vs number of molecules
    
#     Args:
#         df (pd.DataFrame): Data
#         time_step_filter (list): List of time steps to include (None for all)
#         output_dir (str): Output directory for plots
#     """
#     plt.figure(figsize=(12, 8))
    
#     # Filter data if specified
#     if time_step_filter:
#         df_filtered = df[df['time_step'].isin(time_step_filter)]
#         title_suffix = f" (Time Steps: {time_step_filter})"
#         filename_suffix = f"_filtered_{'_'.join(map(str, time_step_filter))}"
#     else:
#         df_filtered = df
#         title_suffix = " (All Time Steps)"
#         filename_suffix = "_all"
    
#     time_steps = sorted(df_filtered['time_step'].unique())
#     colors = plt.cm.coolwarm(np.linspace(0, 1, len(time_steps)))
    
#     for i, ts in enumerate(time_steps):
#         subset = df_filtered[df_filtered['time_step'] == ts].sort_values('molecules')
        
#         if len(subset) > 0 and 'std_mse' in subset.columns:
#             plt.plot(subset['molecules'], subset['std_mse'],
#                     marker='D', linestyle='-', linewidth=2, markersize=6,
#                     color=colors[i], label=f'Time Step {ts}', alpha=0.8)
    
#     plt.xlabel('Number of Molecules')
#     plt.ylabel('Standard Deviation of MSE')
#     plt.title(f'MSE Standard Deviation vs Number of Molecules{title_suffix}')
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.grid(True, alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
    
#     # Save plot
#     Path(output_dir).mkdir(exist_ok=True)
#     plt.savefig(f'{output_dir}/mse_std_vs_molecules{filename_suffix}.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved: {output_dir}/mse_std_vs_molecules{filename_suffix}.png")

# def generate_summary_stats(df):
#     """Generate and print summary statistics"""
#     print("\n" + "="*60)
#     print("DATA SUMMARY STATISTICS")
#     print("="*60)
    
#     print(f"Total data points: {len(df)}")
#     print(f"Molecule counts: {sorted(df['molecules'].unique())}")
#     print(f"Time steps: {sorted(df['time_step'].unique())}")
    
#     if 'mse' in df.columns:
#         print(f"\nMSE Statistics:")
#         print(f"  Mean MSE: {df['mse'].mean():.6f}")
#         print(f"  Median MSE: {df['mse'].median():.6f}")
#         print(f"  Min MSE: {df['mse'].min():.6f}")
#         print(f"  Max MSE: {df['mse'].max():.6f}")
#         print(f"  Std MSE: {df['mse'].std():.6f}")
    
#     if 'mean_computation_time' in df.columns:
#         print(f"\nComputation Time Statistics:")
#         print(f"  Mean Time: {df['mean_computation_time'].mean():.4f} seconds")
#         print(f"  Median Time: {df['mean_computation_time'].median():.4f} seconds")
#         print(f"  Min Time: {df['mean_computation_time'].min():.4f} seconds")
#         print(f"  Max Time: {df['mean_computation_time'].max():.4f} seconds")
    
#     print("\n" + "="*60)

# def main(csv_file_path, output_dir='plots', time_step_filter=None):
#     """
#     Main function to create all MSE analysis plots
    
#     Args:
#         csv_file_path (str): Path to CSV file containing MSE data
#         output_dir (str): Directory to save plots
#         time_step_filter (list): Time steps to include in std deviation plot
#     """
#     print("Starting MSE Analysis and Plotting...")
#     print(f"Input file: {csv_file_path}")
#     print(f"Output directory: {output_dir}")
    
#     try:
#         # Load and validate data
#         df = load_and_validate_data(csv_file_path)
        
#         # Generate summary statistics
#         generate_summary_stats(df)
        
#         # Set up plot styling
#         setup_plot_style()
        
#         # Create output directory
#         Path(output_dir).mkdir(exist_ok=True)
        
#         # Generate all plots
#         print("\nGenerating plots...")
        
#         # 1. MSE vs Molecules (log-log)
#         print("Creating MSE vs Molecules plot...")
#         plot_mse_vs_molecules(df, output_dir)
        
#         # 2. MSE vs Time Step (log-log)
#         print("Creating MSE vs Time Step plot...")
#         plot_mse_vs_timestep(df, output_dir)
        
#         # 3. Computation Time vs Molecules
#         if 'mean_computation_time' in df.columns:
#             print("Creating Computation Time vs Molecules plot...")
#             plot_computation_time_vs_molecules(df, output_dir)
        
#         # 4. MSE Bar Plot (grouped)
#         print("Creating MSE bar plot...")
#         plot_mse_bar_grouped(df, output_dir)
        
#         # 5. MSE Standard Deviation vs Molecules
#         if 'std_mse' in df.columns:
#             print("Creating MSE Standard Deviation plot...")
#             plot_mse_std_vs_molecules(df, time_step_filter, output_dir)
        
#         print(f"\nAll plots saved successfully to '{output_dir}' directory!")
#         print("Analysis complete!")
        
#     except Exception as e:
#         print(f"Error during analysis: {str(e)}")
#         raise

# if __name__ == "__main__":
#     # Example usage
#     csv_file = "/home/nguyt16@ds.vanderbilt.edu/HIPASim/hipa/tests/figs/single_cylinder_validation_2025-09-04_10-17-57/single_cylinder_validation_summary_a0.5_D02.0_threshold0.05.csv"  # Replace with your CSV file path
#     output_directory = "mse_plots"
    
#     # Optional: Filter specific time steps for standard deviation plot
#     # time_steps_to_include = [1, 2, 5, 10]  # Uncomment and modify as needed
#     time_steps_to_include = None
    
#     # Run analysis
#     main(csv_file, output_directory, time_steps_to_include)
    
#     # Additional example with filtered time steps
#     # main(csv_file, "mse_plots_filtered", [1, 5, 10])
