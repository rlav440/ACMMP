import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from pathlib import Path
import tqdm
import pandas as pd
import seaborn as sns
import re


cam_number = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# I have cam number -> scan_number
# should do scan_number -> cam_number

loc = Path("../recons/programmatic_run0")

data_array = np.zeros((2, 22, 9, 10))

dir = 0
ds = {}

for recon in tqdm.tqdm(list(loc.iterdir())):

    cam_n = int(recon.parts[-1].split('_')[1])
    scan_n = int(re.findall(r"\d+", recon.parts[-1])[0])

    if not scan_n in ds:
        ds[scan_n] = dir
        dir += 1

    no_prior = np.genfromtxt(recon / "acmmp_no_prior.txt")
    prior = np.genfromtxt(recon / "acmmp_boosted.txt")

    # datum = datum0/datum1 - 1

    data_array[0, ds[scan_n], cam_n - 2, :] = no_prior
    data_array[1, ds[scan_n], cam_n - 2, :] = prior


def draw_data(data, title, **kwargs):
    x = pd.DataFrame(data)
    x.columns = [str(f) for f in cam_number[:data.shape[1]]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(111, frameon=False, sharex=ax1, sharey=ax1)

    props = {
        'boxprops':{'facecolor':'none'}
    }

    a = x.T
    a.columns = [str(f) for f in range(22)]


    #grab the first result
    inds = np.argsort(data.T[-1, :])
    map = np.zeros_like(inds)
    map[inds] = np.arange(0, inds.shape[0])#use the middle

    t = pd.DataFrame()
    t["recondata"] = data.T.flatten()
    d1, d0 = np.meshgrid(map, cam_number[:data.shape[1]])
    t['cam_name'], t['scan_name'] = d0.flatten(), d1.flatten()

    cmap = plt.get_cmap("coolwarm")




    sns.set_theme("paper", style='ticks')
    # ax1.plot(x.T, color=(0, 0, 0, 0.1), zorder=2)
    sns.boxplot(data=x, palette='gray', color="0.1", zorder=1, ax=ax1, showfliers=False, **props)
    sns.stripplot(data=t, x="cam_name", y="recondata", hue="scan_name",
                  palette='crest',
                  legend=False)


    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax1.set_xlabel("Number of cameras")

    limx = 2 * (max(np.mean(data[0]), np.mean(data[-1])))
    if "lims" in kwargs:
        ax1.set_ylim(kwargs.get("lims"))
    ax1.axhline(0, c="r")
    if "ylabel" in kwargs:
        ax1.set_ylabel(kwargs["ylabel"])
    else:
        ax1.set_ylabel("Relative number of recovered points (%)")

    ax1.set_title(title)
    plt.show()

#
# draw_data(data_array[1, :, :, 4] * 100, "Percent points with completeness < 2 mm, using prior", lims=[0, 100])
draw_data(
    (data_array[1, :, :5, 3] / data_array[0, :, :5, 3] - 1) * 100,
    "Percent improvement in  < 0.5 mm completeness using a prior.",
    lims=[-10, 100],
    ylabel="Percent improvement in number of points below threshold."
)
# draw_data(data_acc2, "Percent improvement in accuracy below 2 mm", lims=[0, 100])
# draw_data(data_acc10, "Percent improvement in accuracy below 10 mm", lims=[0, 100])
#
#
#
# draw_data(data_cmp05, "Percent improvement in completeness below 0.5 mm", lims=[0, 100])
# draw_data(data_cmp2, "Percent improvement in completeness below 2 mm", lims=[0, 100])
# draw_data(data_cmp10, "Percent improvement in completeness below 10 mm", lims=[0, 100])
#
#
# draw_data(data_acc_median, "Median Accuracy", lims=[0, 0.5], ylabel="Median Accuracy (mm)")
# draw_data(data_cmp_median, "Median Completeness", lims=[0, 5], ylabel="Median Completeness (mm)")
#
# fix, ax = plt.subplots(1,1)
#
# ax.violinplot(dataset=data, positions=cam_number)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# ax.set_ylabel("Relative number of recovered points (%)")
# ax.set_xlabel("Number of cameras")
#
# ax.set_ylim([-10, 50])
# ax.axhline(0, c="r")
# plt.show()
