import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_graph(files, title, filename):
    x_axis_val = 'arg_domain_size'
    y_axis_val = 'time'
    transfer_time_val = 'transfer_time'
    transfer_suffix = ' + transfer time'

    dfs = []
    transfer_bars = 0
    for label, file in files.items():
        df = pd.read_csv(os.path.join(script_dir, f'data/{file}'))

        original_df = df[[x_axis_val, y_axis_val]].copy()
        original_df = original_df.rename(columns={y_axis_val: f'{label}'})
        dfs.append(original_df)

        if transfer_time_val in df.columns:
            adjusted_df = df[[x_axis_val, y_axis_val, transfer_time_val]].copy()
            adjusted_df[label + transfer_suffix] = adjusted_df[y_axis_val] + adjusted_df[transfer_time_val]
            adjusted_df = adjusted_df[[x_axis_val, label + transfer_suffix]]
            dfs.append(adjusted_df)
            transfer_bars += 1

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=x_axis_val)

    # Plotting
    x_vals = merged_df[x_axis_val]
    x = np.arange(len(x_vals))
    bar_width = 0.2


    plt.figure(figsize=(10, 6))
    for i, label in enumerate(merged_df.columns[1:]):
        plt.bar(x + i * bar_width, merged_df[label], width=bar_width, label=label)

        if transfer_time_val in df.columns:
            plt.bar(x + i * bar_width, merged_df[label + transfer_suffix], width=bar_width, label=label + transfer_suffix, alpha=0.5)

    plt.xlabel('Argument Domain Size')
    plt.ylabel('Computation Time (ms)')
    plt.title(title)

    plt.xticks(x + bar_width * (len(files) + transfer_bars - 1) / 2, x_vals)

    plt.yscale('log') 
    plt.gca().set_ylim(bottom=0.001)

    plt.legend()
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'bar_chart_{filename}_{timestamp}'

    # Save the plot
    plt.savefig(os.path.join(script_dir, f'figures/{filename}.png'))


def generate_graph_from_single_file(file, x_axis_val, title, filename, label_map=None):
    df = pd.read_csv(os.path.join(script_dir, f'data/{file}'))
    exclude_list = ['output_size']
    plot_columns = [col for col in df.columns if col not in exclude_list and col != x_axis_val and pd.api.types.is_numeric_dtype(df[col])]

    if not plot_columns:
        raise ValueError("No numeric columns to plot.")

    x_vals = df[x_axis_val]
    x = np.arange(len(x_vals))
    bar_width = 0.8 / len(plot_columns)

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(plot_columns):
        plt.bar(x + i * bar_width, df[col], width=bar_width, label=label_map[col])

    plt.xlabel(x_axis_val.replace('_', ' ').title())
    plt.ylabel('Value')
    plt.title(title)

    plt.xticks(x + bar_width * (len(plot_columns) - 1) / 2, x_vals)
    plt.yscale('log')
    plt.gca().set_ylim(bottom=0.001)

    plt.legend()
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'bar_chart_{filename}_{timestamp}'

    plt.savefig(os.path.join(script_dir, f'figures/{filename}.png'))


if __name__ == '__main__':
    # Delete all previous figures
    for filename in os.listdir(os.path.join(script_dir, 'figures')):
        file_path = os.path.join(script_dir, 'figures', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    generate_graph({
        '1 row per thread': 'gpu_thread_1.csv',
        '2 rows per thread': 'gpu_thread_2.csv',
        '4 rows per thread': 'gpu_thread_4.csv',
        '8 rows per thread': 'gpu_thread_8.csv',
    }, 'GPU Performance against Argument Domain Size', 'gpu')

    generate_graph({
        'GPU computation time': 'gpu_thread_1.csv',
        'CPU computation time': 'cpu_2_states.csv',
    }, 'GPU vs CPU Performance against Argument Domain Size', 'gpu_vs_cpu')

    generate_graph({
        'GPU computation time': 'gpu_transfer.csv',
        'CPU computation time': 'cpu_2_states.csv',
    }, 'GPU vs CPU Performance against Argument Domain Size', 'gpu_mem_vs_cpu')

    generate_graph_from_single_file(
        'gpu_transfer.csv',
        'arg_domain_size',
        'GPU and Transfer Latency against Argument Domain Size',
        'gpu_mem_latency', {
            'in_transfer': 'Inputs Transfer Time',
            'out_transfer': 'Output Transfer Time',
            'transfer_time': 'Total Transfer Time',
            'time': 'Computation Time',
        })