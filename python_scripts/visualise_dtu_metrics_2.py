import pyvista as pv
from matplotlib import pyplot as plt
from pathlib import Path
import tqdm
import pandas as pd
import seaborn as sns
import re
import numpy as np

from abiStereoRaySampler.DTU_eval import dtu_eval


cam_number = [2, 3, 5, 9]

# I have cam number -> scan_number
# should do scan_number -> cam_number

loc = Path("../recons/programmatic_run1")


cloud0 = Path("../recons/programmatic_run1/scan1_5_cam/acmmp_boost_1.ply")
cloud1 = Path("../recons/programmatic_run1/scan1_5_cam/ACMMP_no_prior.ply")
cloud2 = Path("../recons/programmatic_run1/scan1_5_cam/ACMMP_x2.ply.ply")
ground_truth = Path("/home/rlav440/CLionProjects/gipuma/scripts/data/dtu/SampleSet/MVS Data/Points/stl/stl001_total.ply")
# ply
# dtu_eval(
#     np.array(pv.read(ground_truth).points),
#     np.array(pv.read(cloud0).points),
#     np.array(pv.read(cloud1).points),
# )
#
#
#


data_array = np.zeros((3, 22, 4, 18))

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
    augmented = np.genfromtxt(recon / "acmmp_x2.txt")

    # datum = datum0/datum1 - 1
    c = cam_number.index(cam_n)
    data_array[0, ds[scan_n], c, :] = no_prior * 100
    data_array[1, ds[scan_n], c, :] = prior * 100
    data_array[2, ds[scan_n], c, :] = augmented * 100



def draw_data(data, title, **kwargs):
    x = pd.DataFrame(data)
    x.columns = [str(f) for f in cam_number[:data.shape[1]]]

    if "ax" in kwargs:
        ax = kwargs['ax']
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
    sns.boxplot(data=x, palette='gray', color="0.1", zorder=1, ax=ax, showfliers=False, **props)
    sns.stripplot(data=t, x="cam_name", y="recondata", hue="scan_name", ax=ax,
                  palette='crest',
                  legend=False)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ax.set_xlabel("Number of cameras")

    limx = 2 * (max(np.mean(data[0]), np.mean(data[-1])))
    if "lims" in kwargs:
        ax.set_ylim(kwargs.get("lims"))
    ax.axhline(0, c="r")
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])
    else:
        ax.set_ylabel("Relative number of recovered points (%)")

    ax.set_title(title)
    if not "ax" in kwargs:
        plt.show()

fig, ax = plt.subplots(1,3)
label = 'Points within 0.2mm completeness (%)'
data_num = 6
draw_data(
    data_array[0, :, :, data_num],
    "Base method.",
    lims=[0, 100],
    ylabel=label,
    ax=ax[0],
    )

draw_data(
    data_array[2, :, :, data_num],
    "double fusion.",
    lims=[0, 100],
    ylabel=label,
    ax=ax[1],
)


draw_data(
    data_array[1, :, :, data_num],
    "Prior + double fusion.",
    lims=[0, 100],
    ylabel=label,
    ax=ax[2],
)


fig, ax = plt.subplots(1,2)
# draw_data(
#     data_array[0, :, :, 4],
#     "Percent improvement in  < 0.5 mm completeness using double fusion.",
#     lims=[-10, 100],
#     ylabel="Percent improvement in number of points below threshold.",
#     ax=ax[0],
# )
# draw_data(
#     (data_array[1, :, :, 4] / data_array[2, :, :, 4] - 1) * 100,
#     "Percent improvement in  < 0.5 mm completeness using a prior over double fusion.",
#     lims=[-10, 100],
#     ylabel="Percent improvement in number of points below threshold.",
#     ax=ax[1],
#     )
# plt.show()

draw_data(
    (data_array[2, :, :, 4] / data_array[0, :, :, 4] - 1) * 100,
    "Percent improvement in  < 0.5 mm completeness using double fusion.",
    lims=[-10, 100],
    ylabel="Percent improvement in number of points below threshold.",
    ax=ax[0],
)

draw_data(
    (data_array[1, :, :, 4] / data_array[2, :, :, 4] - 1) * 100,
    "Percent improvement in  < 0.5 mm completeness using a prior over double fusion.",
    lims=[-10, 100],
    ylabel="Percent improvement in number of points below threshold.",
    ax=ax[1],
    )
plt.show()

fig, ax = plt.subplots(1,2)

draw_data(
    data_array[2, :, :, 4],
    "0.5 mm completeness run twice.",
    lims=[-10, 100],
    ylabel="Percent of points meeting the threshold.",
    ax=ax[0]
)

draw_data(
    data_array[0, :, :, 4],
    "0.5 mm completeness, with a default implementation.",
    lims=[-10, 100],
    ylabel="Percent of points meeting the threshold.",
    ax=ax[1]
)

plt.show()

# plt.figure()
# plt.plot(cam_number, data_array[0,0,:,6], label="Default")
# plt.plot(cam_number, data_array[1,0,:,6], label="W/prior")
# plt.plot(cam_number, data_array[2,0,:,6], label='w/dfusion')
# plt.legend()
# plt.figure()
# plt.plot(cam_number, data_array[0,1,:,6], label="Default")
# plt.plot(cam_number, data_array[1,1,:,6], label="W/prior")
# plt.plot(cam_number, data_array[2,1,:,6], label='w/dfusion')
# plt.legend()
# plt.show()
