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

loc = Path("../recons/dtu_eval")


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
# acc05, acc2,acc5, acc10, cmp05, cmp2, cmp5, cmp10, acc_mean, acc_median, completeness_mean, completeness_median
n_camarange = len(cam_number)
num_folders = len(list(loc.iterdir()))
n_points = int( np.ceil(num_folders / n_camarange))

n_datum = 12 # amount of successful datasets
n_dataset = 5
data_array = np.zeros((n_dataset, n_points, n_camarange, n_datum)) * np.NaN

dir = 0
ds = {}
try:
    for recon in tqdm.tqdm(list(loc.iterdir())):

        if not (recon/'acmmp_boosted.txt').exists():
            continue

        cam_n = int(recon.parts[-1].split('_')[1])
        scan_n = int(re.findall(r"\d+", recon.parts[-1])[0])

        if not scan_n in ds:
            ds[scan_n] = dir
            dir += 1

        #why not just do this

        no_prior = np.genfromtxt(recon / "acmmp_no_prior.txt")
        prior_df = np.genfromtxt(recon / "acmmp_boosted.txt")
        augmented_df = np.genfromtxt(recon / "acmmp_x2.txt")

        prior_boostrap = np.genfromtxt(recon/'acmmp_boost_single.txt')
        prior_full = np.genfromtxt(recon/'ACMMP_full_prior.txt')


        # datum = datum0/datum1 - 1
        c = cam_number.index(cam_n)
        data_array[0, ds[scan_n], c, :] = no_prior * 100
        data_array[1, ds[scan_n], c, :] = prior_df * 100
        data_array[2, ds[scan_n], c, :] = augmented_df * 100
        data_array[3, ds[scan_n], c, :] = prior_boostrap * 100
        data_array[4, ds[scan_n], c, :] = prior_full * 100

except:
    pass


np.save( 'good_evaluation_data.npy', data_array,)
raise ValueError('finished evaluating')
data_array = np.load("old_method_evaluation_data.npy")
def draw_data(data, title, **kwargs):

    data = data[~np.any(np.isnan(data), axis=1)]

    l = len(data)

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
    a.columns = [str(f) for f in range(l)]


    #grab the first result
    inds = np.argsort(data.T[-1, :])
    map = np.zeros_like(inds)
    map[inds] = np.arange(0, inds.shape[0]) #use the middle

    t = pd.DataFrame()
    t["recondata"] = data.T.flatten()
    d1, d0 = np.meshgrid(map, cam_number[:data.shape[1]])
    t['cam_name'], t['scan_name'] = d0.flatten(), d1.flatten()

    #get all values where no Na in dataframe

    # t = t[~t.isnull().any(axis=0)]

    cmap = plt.get_cmap("coolwarm")


    sns.set_theme("paper", style='ticks')
    # ax1.plot(x.T, color=(0, 0, 0, 0.1), zorder=2)
    sns.boxplot(data=x, palette='gray', color="0.1", zorder=1, ax=ax, showfliers=False, **props)
    sns.stripplot(data=t, x="cam_name", y="recondata", hue="scan_name", ax=ax,
                  palette=cmap,
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


def draw_plot_series(data: list[np.ndarray], name_labels:list[str], label_x, label_y, **kwargs):

    # get the data
    # chuck it into a dataframe
    # plot using a seaborn catplot


    data = [d[~np.any(np.isnan(d), axis=1)] for d in data]
    n = [i * np.ones_like(d) for i, d in enumerate(data)]
    data = np.concatenate(data, axis=0)
    n = np.concatenate(n, axis=0)
    n = n.T.flatten().astype(int)

    nl = [name_labels[nd] for nd in n]

    l = len(data)
    x = pd.DataFrame(data)
    x.columns = [str(f) for f in cam_number[:data.shape[1]]]

    if "ax" in kwargs:
        ax = kwargs['ax']
    else:
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)


    a = x.T
    a.columns = [str(f) for f in range(l)] #this is the number of cameras
    #grab the first result
    inds = np.argsort(data.T[-1, :])
    map = np.zeros_like(inds)
    map[inds] = np.arange(0, inds.shape[0]) #use the middle

    t = pd.DataFrame()
    t[label_y] = data.T.flatten()
    d1, d0 = np.meshgrid(map, cam_number[:data.shape[1]])
    t[label_x], t['scan_name'] = d0.flatten(), d1.flatten()
    t['Method'] = nl


    props = {
        'boxprops':{'facecolor':((.4, .6, .8, .5))}
    }

    sns.boxplot(data=t,
                x=label_x,
                y=label_y,
                hue='Method',
                dodge=True,
                  # width=0.5,
                  palette="YlGnBu_d",
                  # saturation=0.5,
                whis=10000000000,
                  )
    sns.despine(offset=2, trim = True)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    ax.axhline(0, c='r')
    plt.show()


# get the average data for the outputs
dp = 11









data_num = dp
s = 100

l = "Mean completeness (mm)"
a =data_array[0, ..., data_num]/s
a = 1
# draw_plot_series(
#     data = [
#         data_array[2, ..., data_num]/s/a,
#         data_array[3, ..., data_num]/s/a,
#         data_array[1, ..., data_num]/s/a,
#         ],
#     name_labels=["Default", "Set Fusion", "Bootstrapped Prior", "SD + BP"],
#     label_x = 'Number of Cameras',
#     label_y = l,
# )


averages = np.nanmean(data_array, axis=1) #what is this mean along.


print("Mean of main method completeness")
print(averages[[0,2,-2,1], :, dp]/100)


print("relative Mean performance of proposals")
print(100 * (averages[[2,-2,1], :, dp]/averages[0,:,dp] - 1))


print("bootstrapped prior vs full")
print((averages[[3,4], :, dp])/100)



dp = 11
averages = np.nanmedian(data_array, axis=1) #what is this mean along.

print("Median of median completeness")
print(averages[[0,2,-2,1], :, dp]/100)

print("relative Medianan performance of proposals compared to basline")
print(100 * (averages[[2,-2,1], :, dp]/averages[0,:,dp] - 1))

print("bootstrapped prior vs full")
print((averages[[-2,-1], :, dp])/100)



print("relative performance of proposals")

print(100 * (averages[-2, :, dp]/averages[1,:,dp] - 1))
s = 100
data_num = dp
l = "0.5 mm Completeness"
l = "Percentage change in median completeness from base -> df+p"
draw_plot_series(
    data = [
        100 * ((data_array[-2, ..., data_num]/data_array[0, ..., dp]) - 1),
        ],
    name_labels=["Default"],
    label_x = 'Number of Cameras',
    label_y = l,
)


# raise ValueError("dun wanna")



draw_plot_series(
    data = [
        data_array[0, ..., data_num]/s,
        data_array[-2, ..., data_num]/s,
        data_array[-1, ..., data_num]/s,

    ],
    name_labels=["Default", "Bootstrapped Prior", "Full Prior"],
    label_x = 'Number of Cameras',
    label_y = l,
)





fig, ax = plt.subplots(1,3, sharey=True)
label = 'Points within 0.5mm completeness (%)'

lims = [-10, 100]
draw_data(
    data_array[0, :, :, data_num],
    "Base method.",
    lims=lims,
    ylabel=label,
    ax=ax[0],
    )

draw_data(
    data_array[2, :, :, data_num],
    "Double fusion.",
    lims=lims,
    ylabel=label,
    ax=ax[1],
)


draw_data(
    data_array[1, :, :, data_num],
    "Prior + Double fusion.",
    lims=lims,
    ylabel=label,
    ax=ax[2],
)



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
fig, ax = plt.subplots(1,2, sharey=True)

lims = [-10, 100]
dn = 5

draw_data(
    (data_array[2, :, :, dn] / data_array[0, :, :, dn] - 1) * 100,
    "Percent improvement in  < 0.5 mm completeness using double fusion.",
    lims=lims,
    ylabel="Percent improvement in number of points below threshold.",
    ax=ax[0],
)

draw_data(
    (data_array[1, :, :, dn] / data_array[2, :, :, dn] - 1) * 100,
    "Percent improvement in  < 0.5 mm completeness using a prior over double fusion.",
    lims=lims,
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



fig, ax = plt.subplots(1,2)

draw_data(
    data_array[0, :, :, 9]/100,
    "Median Accuracy of ACMMP.",
    lims=[0, 0.5],
    ylabel="Median accuracy (mm)",
    ax=ax[0]
)

draw_data(
    data_array[1, :, :, 9]/100,
    "Median Accuracy run twice, with a secondary prior.",
    lims=[0, 0.5],
    ylabel="Median accuracy (mm)",
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
