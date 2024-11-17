import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt

def get_psnr_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['ours_30000']['PSNR']

def collect_psnr_data(base_folder, equalize=False, scale_lr=False):
    psnr_data = defaultdict(float)
    for folder_name in os.listdir(base_folder):
        if folder_name.startswith('eval_mv_'):
            if not equalize:
                if folder_name.endswith('_equalized') or folder_name.endswith('_equalized_scaled_lr'):
                    continue
            else:
                if not scale_lr and not folder_name.endswith('_equalized'):
                    continue
                if scale_lr and not folder_name.endswith('_equalized_scaled_lr'):
                    continue

            m = int(folder_name.split('_')[2])  # m: number of views
            if m not in [1, 5, 10, 20]:
                continue
            scenes = os.listdir(os.path.join(base_folder, folder_name))
            scenes = [scene for scene in scenes if os.path.isdir(os.path.join(base_folder, folder_name, scene))]
            for scene in scenes:
                results_file = os.path.join(base_folder, folder_name, scene, 'results.json')
                # print(results_file)
                psnr = get_psnr_from_json(results_file)
                psnr_data[m] += psnr / len(scenes)
    return psnr_data

def plot_psnr(psnr_data_equalize, psnr_data_equalize_scaled_lr):
    xs = sorted(psnr_data_equalize.keys(), key=int)
    ys_equalize = [psnr_data_equalize[x] for x in xs]
    ys_equalize_scaled_lr = [psnr_data_equalize_scaled_lr[x] for x in xs]
    print(xs, ys_equalize)
    print(xs, ys_equalize_scaled_lr)

    # plot data points too
    plt.plot(xs, ys_equalize, label='Default Hypers', marker='o')
    plt.plot(xs, ys_equalize_scaled_lr, label='Scaled Hypers (ours)', marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('PSNR')
    plt.grid()
    # label y values
    for i in range(1, len(ys_equalize)):
        plt.annotate(f'{ys_equalize[i]:.2f}', (xs[i], ys_equalize[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'{ys_equalize_scaled_lr[i]:.2f}', (xs[i], ys_equalize_scaled_lr[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.ylim(24, 28)
    plt.legend()
    plt.title('PSNR vs Batch Size (Garden Scene)')
    plt.savefig('ablation_plots/garden_PSNR_vs_scaled_hypers.png')
    plt.savefig('ablation_plots/garden_PSNR_vs_scaled_hypers.pdf')
    plt.show()
    plt.close()

base_folder = './eval_garden'
psnr_data_equalize = collect_psnr_data(base_folder, equalize=True, scale_lr=False)
psnr_data_equalize_scaled_lr = collect_psnr_data(base_folder, equalize=True, scale_lr=True)

plot_psnr(psnr_data_equalize, psnr_data_equalize_scaled_lr)