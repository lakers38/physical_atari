# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def remove_outliers_zscore(x, threshold=3.0):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return np.ones_like(x, dtype=bool)
    return np.abs(x - mean) < threshold * std


def plot_data(files, title, remove_outliers, output_path, ylabel, xlabel, is_scatter=False, color=None):
    plt.figure(figsize=(10, 6))

    plt.ticklabel_format(style='plain', axis='x')

    for file in files:
        label = os.path.basename(os.path.dirname(file))
        data = np.fromfile(file, dtype=np.float32)

        if is_scatter:
            data = data.reshape(-1, 2)
            x_data, y_data = data[:, 0], data[:, 1]
        else:
            x_data, y_data = np.arange(len(data)), data

        if remove_outliers and len(y_data) > 0:
            mask = remove_outliers_zscore(y_data)
            num_removed = len(y_data) - np.count_nonzero(mask)
            if num_removed > 0:
                print(f"{file}: removed {num_removed} outliers.")
            x_data, y_data = x_data[mask], y_data[mask]

        if is_scatter:
            plt.scatter(x_data, y_data, label=label, alpha=0.6)
        else:
            plt.plot(x_data, y_data, label=label, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, format='jpg')
    print(f"Saved plot: {output_path}")
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="plot_data.py arguments.")
    parser.add_argument('root_dir', type=str, help="path to experiment result directory")
    parser.add_argument('--title', type=str, default=None, help="plot title")
    parser.add_argument('--remove-outliers', action='store_true', help="remove statistical outliers using z-score.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    extensions = {
        '.loss': {
            'ylabel': 'Loss',
            'xlabel': 'Training Step',
            'outfile': 'loss_plot.jpg',
            'is_scatter': False,
            'color': 'red',
        },
        '.score': {
            'ylabel': 'Average Score',
            'xlabel': 'Episode',
            'outfile': 'score_plot.jpg',
            'is_scatter': False,
            'color': None,
        },
        '.scatter': {
            'ylabel': 'Episode Score',
            'xlabel': 'Episode End',
            'outfile': 'scatter_plot.jpg',
            'is_scatter': True,
            'color': None,
        },
    }

    file_map = {ext: [] for ext in extensions}
    for dirpath, _, filenames in os.walk(args.root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)

            _, ext = os.path.splitext(fname)
            if ext in file_map:
                file_map[ext].append(full_path)

    for ext, cfg in extensions.items():
        files = file_map[ext]
        if not files:
            continue
        plot_data(
            files=files,
            title=args.title or f"{cfg['ylabel']} Plot",
            remove_outliers=args.remove_outliers,
            output_path=os.path.join(args.root_dir, cfg['outfile']),
            ylabel=cfg['ylabel'],
            xlabel=cfg['xlabel'],
            is_scatter=cfg['is_scatter'],
            color=cfg['color'],
        )

    print("Complete.")
