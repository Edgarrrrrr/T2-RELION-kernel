import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Process kernel test results and plot speed-up / error figures")
parser.add_argument("--input",  default="./kernel_test_results.csv",
                    help="CSV file produced by kernel tests")
parser.add_argument("--fig7",   default="./Fig7.pdf",
                    help="Output filename for speed-up figure")
parser.add_argument("--fig8",   default="./Fig8.pdf",
                    help="Output filename for error-boxplot figure")
args = parser.parse_args()

kernel_result_csv   = args.input
figure_7_outfile    = args.fig7
figure_8_outfile    = args.fig8

# import matplotlib.font_manager as fm
# fm._load_fontmanager(try_read_cache=False)
# font_path = '/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf'
# prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = prop.get_name()

# Read data and preprocess
df = pd.read_csv(kernel_result_csv)

# Split 'iteration' and 'data_num' from 'data name'
df[['iteration', 'data_num']] = df['data name'].str.split('_', expand=True)
df['iteration'] = df['iteration'].astype(int)
df['data_num'] = df['data_num'].astype(int)

# Extract data for all optimization versions
optimized_data = []
for idx, row in df.iterrows():
    # Optimized version data extraction
    for i in range(6):
        desc_col = f'optimized description.{i}' if i > 0 else 'optimized description'
        time_col = f'optimized times (us).{i}' if i > 0 else 'optimized times (us)'
        error_col = f'avg rel error.{i}' if i > 0 else 'avg rel error'
        
        optimized_data.append({
            'data name': row['data name'],
            'iteration': row['iteration'],
            'data_num': row['data_num'],
            'optimization_description': row[desc_col],
            'time_us': row[time_col],
            'avg_rel_error': row[error_col]
        })

optimized_df = pd.DataFrame(optimized_data)

# Extract trans_num, orientation_num, and image_size for each iteration, record maximum and minimum values.
data_size = {}
for idx, row in df.iterrows():
    iteration = row['iteration']
    data_num = row['data_num']
    orientation_num = row['orientation num']
    translation_num = row['translation num']
    image_size = row['image size']
    
    if iteration in data_size:
        data_size[iteration]['orientation_num_max'] = max(data_size[iteration]['orientation_num_max'], orientation_num)
        data_size[iteration]['orientation_num_min'] = min(data_size[iteration]['orientation_num_min'], orientation_num)
        data_size[iteration]['translation_num_max'] = max(data_size[iteration]['translation_num_max'], translation_num)
        data_size[iteration]['translation_num_min'] = min(data_size[iteration]['translation_num_min'], translation_num)
        data_size[iteration]['image_size_max'] = max(data_size[iteration]['image_size_max'], image_size)
        data_size[iteration]['image_size_min'] = min(data_size[iteration]['image_size_min'], image_size)
    else:
        data_size[iteration] = {
            'orientation_num_max': orientation_num,
            'orientation_num_min': orientation_num,
            'translation_num_max': translation_num,
            'translation_num_min': translation_num,
            'image_size_max': image_size,
            'image_size_min': image_size
        }


# Ensure 'optimization_description' maintains a specific order
optimization_order = ['original', 'Im2col Multi-Level Blocking + CUDA Core',
                      'Im2col Multi-Level Blocking + Tensor Core', '+ Conflict Removal',
                      '+ Register-Based Texture Fetch Masking', '+ Collaborative Thread-Block-Level Data Reuse']  # replace with your actual order
optimized_df['optimization_description'] = pd.Categorical(optimized_df['optimization_description'], categories=optimization_order, ordered=True)

# Calculate the original times as the mean for each iteration
original_times = optimized_df[optimized_df['optimization_description'] == 'original'].groupby('iteration')['time_us'].mean().rename('original_time')

# Merge the original times with the optimized data
optimized_df = optimized_df.merge(original_times, on='iteration')

# For each iteration, same optimization description, different data, calculate the mean
optimized_df_mean = optimized_df.groupby(['iteration', 'optimization_description']).agg({
    'time_us': 'mean',
    'avg_rel_error': 'mean',
    'original_time': 'mean'
}).reset_index()

# Ensure the optimization_description retains the order after grouping
optimized_df_mean['optimization_description'] = pd.Categorical(optimized_df_mean['optimization_description'], categories=optimization_order, ordered=True)

# Save as CSV
# optimized_df_mean.to_csv('optimized_df_mean.csv', index=False)

# Recalculate the speedup based on the mean "original" time per iteration
optimized_df_mean['speedup'] = optimized_df_mean['original_time'] / optimized_df_mean['time_us']

# save as csv
# optimized_df_mean.to_csv('optimized_df_mean.csv', index=False)

# Recalculate the speedup based on the mean "original" time per iteration
optimized_df_mean['speedup'] = optimized_df_mean['original_time'] / optimized_df_mean['time_us']

custom_palette = [(38/255, 70/255, 83/255, 1), 
                  (70/255, 120/255, 142/255, 1), 
                  (120/255, 183/255, 201/255, 1), 
                  (229/255, 139/255, 123/255, 1),
                  (151/255, 179/255, 25/255, 1),
                  (246/255, 224/255, 147/255, 1),
                  ]

# Set up subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=False)
plt.subplots_adjust(hspace=0.2)

# Define iteration ranges and corresponding y-limits
groups = [
    {'iter_range': (1, 8), 'y_lim': (0, 4)},
    {'iter_range': (9, 14), 'y_lim': (0, 3.5)},
    {'iter_range': (15, 19), 'y_lim': (0, 32)}
]

# Create the plots
for ax, group in zip(axes, groups):
    start, end = group['iter_range']
    mask = (optimized_df_mean['iteration'] >= start) & (optimized_df_mean['iteration'] <= end)
    group_df = optimized_df_mean[mask]
    
    # Plot bar chart
    sns.barplot(
        data=group_df,
        x='iteration',
        y='speedup',
        hue='optimization_description',
        palette=custom_palette,
        ax=ax,
        estimator=np.mean,
        edgecolor='black'
    )
    
    # Set axis labels and limits
    ax.set_ylim(group['y_lim'])
    ax.set_xlabel('Iteration', fontsize=19)
    ax.set_ylabel('Speedup', fontsize=19)
    ax.tick_params(axis='both', labelsize=15)
    
    # Add data labels above bars
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    # height * 1.05,  # Adjusted label position
                    group['y_lim'][-1] * 0.03 + height,
                    f'{height:.2f}x',
                    ha='center',
                    va='bottom',
                    fontsize=15,
                    rotation=90
                )

# Fix legend layout (3 rows, 2 columns)
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    # bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    fontsize=17
)

# Remove legends from individual subplots
for ax in axes:
    ax.get_legend().remove()

# Save and display the figure
plt.savefig(figure_7_outfile, bbox_inches='tight', format='pdf', dpi=300)
plt.clf()

# Error plot – keep only original and the last optimization version
optimized_df_err = optimized_df[optimized_df['optimization_description'].isin(['original', optimization_order[-1]])]

optimized_df_err['optimization_description'] = optimized_df_err['optimization_description'].replace(
    {'original': 'original', optimization_order[-1]: 'optimized'}  # Replace 'other_value' with the actual name of the value you want to change
)

# optimized_df_err.to_csv('optimized_df_err.csv', index=False)

plt.figure(figsize=(15, 8))
# plt.tick_params(axis='both', labelsize=19) 
ax = sns.boxplot(data=optimized_df_err, x='iteration', y='avg_rel_error', 
            hue='optimization_description', showfliers=False, 
            palette=[custom_palette[1], custom_palette[-1]], 
            hue_order=['original', 'optimized'],
            width=0.6, dodge=True)
ax.tick_params(axis='both', labelsize=21)

# Print mean and quartiles for original and last optimization versions
print("\nError Distribution Comparison:")
origin_avg_IQR = 0.
optimized_avg_IQR = 0.
for iteration in optimized_df_err['iteration'].unique():
    iteration_data = optimized_df_err[optimized_df_err['iteration'] == iteration]
    original_error = iteration_data[iteration_data['optimization_description'] == 'original']['avg_rel_error']
    optimized_error = iteration_data[iteration_data['optimization_description'] == optimized_df_err['optimization_description'].unique()[-1]]['avg_rel_error']
    
    original_mean = original_error.mean()
    optimized_mean = optimized_error.mean()
    original_q1 = original_error.quantile(0.25)
    optimized_q1 = optimized_error.quantile(0.25)
    
    original_q3 = original_error.quantile(0.75)
    optimized_q3 = optimized_error.quantile(0.75)
    origin_avg_IQR += original_q3 - original_q1
    optimized_avg_IQR += optimized_q3 - optimized_q1
    print(f"\nIteration {iteration}:")
    print(f"  Original Mean Error = {original_mean:.6e}, 1/4 Quantile = {original_q1:.6e} 3/4 Quantile = {original_q3:.6e} IQR = {original_q3 - original_q1:.6e}")
    print(f"  Optimized Mean Error = {optimized_mean:.6e}, 1/4 Quantile = {optimized_q1:.6e} 3/4 Quantile = {optimized_q3:.6e} IQR = {optimized_q3 - optimized_q1:.6e}")
    print(f"  Optimized/Original = {optimized_mean/original_mean:.6f}, IQR = {optimized_q3 - optimized_q1:.6e}/{original_q3 - original_q1:.6e} = {optimized_mean/original_mean:.6f}x")

origin_avg_IQR /= len(optimized_df_err['iteration'].unique())
optimized_avg_IQR /= len(optimized_df_err['iteration'].unique())
print(f"\nAverage IQR for Original: {origin_avg_IQR:.6e}")
print(f"Average IQR for Optimized: {optimized_avg_IQR:.6e}")

plt.yscale('log')

from matplotlib.ticker import FuncFormatter
# def sci_plain(y, _):
#     if y == 0: return "0"
#     a, b = f"{y:.0e}".split("e") 
#     return f"{a}e{int(b)}"

# ax.yaxis.set_major_formatter(FuncFormatter(sci_plain))
ax.set_yticks([3e-7, 2e-7, 1e-7, 6e-8])

ax.tick_params(axis='y', labelsize=21)

plt.xlabel('Iteration',fontsize=21)
plt.ylabel('Average Relative Error', fontsize=21)
# plt.tick_params(axis='y', labelsize=25)

# Add horizontal line for FP32 machine precision
float_precision = 1.19209290E-07
plt.axhline(y=float_precision, color='r', linestyle='--', label='FP32 machine precision')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=23)
plt.tight_layout()
plt.savefig(figure_8_outfile, format='pdf', dpi=300)
