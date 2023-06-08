import pyvista as pv
from matplotlib import pyplot as plt
from pathlib import Path
import tqdm
import pandas as pd
import seaborn as sns
import re
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
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
        data_array[0, ds[scan_n], c, :] = no_prior
        data_array[1, ds[scan_n], c, :] = augmented_df
        data_array[2, ds[scan_n], c, :] = prior_boostrap
        data_array[3, ds[scan_n], c, :] = prior_df

        data_array[4, ds[scan_n], c, :] = prior_full
except:
    pass


# what do we need to do here
selected_metric = -1




# look at the mean ocmpletness of the data
data = data_array[:,:,:, selected_metric]
# data = data[~np.any(np.isnan(data), axis=1)]

method_number = np.ones_like(data) * np.array([0,1,2,3,4])[:, None, None]
cam_number = np.ones_like(data) * np.array([2, 3, 5, 9])[None, None, :]
scan_number = np.ones_like(data) * np.arange(0,120)[None, :, None]


t = pd.DataFrame()
t["recondata"] = data.T.flatten()
t['cam_name'] = cam_number.flatten()
t['scan_name'] = scan_number.flatten()
t['method_number'] = method_number.flatten()

method_dict = {
    0:"ACMMP", 1:"DF + P", 2:'DF', 3:"P", 4:"FP"
}

unique_methods = t['method_number'].unique()
methods = [1,2,3,4]


d_mat = np.zeros((4, 5, 5))
s_mat = -np.ones((4, 5, 5))

for method in methods:
    base = t[t['method_number']==0]
    t2 = t[t['method_number']==method]
    unique_cams = t2['cam_name'].unique()
    fig, axs = plt.subplots(1,4)


    for i, cam in enumerate(unique_cams):
        # stats.probplot(
        #     # t2[t2['cam_name']==cam]['recondata'],
        #     data[method,:, i] - data[0,:, i],
        #     dist='norm', plot=axs[i],
        # )
        # axs[i].hist(data[method,:, i] - data[0,:, i])
        # axs[i].set_title(f"{int(cam)} Cameras")
        for x in range(method):
            if x == method:
                continue


            diffs = data[method, :, i] - data[x, :, i]
            arg_inds = np.argsort(diffs)
            # s_value = np.ceil(0.025 * len(diffs))
            # maks = arg_inds[int(s_value):int(-s_value)]
            maks = arg_inds

            test_result = stats.ttest_rel(data[method,maks, i], data[x,maks, i], nan_policy='omit')
            d_mat[i, method, x] = np.nanmean(data[method, maks, i] - data[x, maks, i])
            s_mat[i, method, x] = test_result.pvalue


        # axs[i].set_xlim([-2,2])
    # fig.suptitle(f"Probability plots for method {method_dict[method]}")
    # plt.show()

# what are the results that we are seeing: these metrics are not normally distributed


for id, cam in enumerate([2,3,5,9]):
    print(f"Results for camera {cam}")
    print("#########################################")

    #grab the comparisons
    comps = s_mat[id, 1:, 0]
    reject_null, corrected_p, _, _ = multipletests(comps, alpha=0.05)

    # what comparisons do we want to see
    print("First order comparisons to baseline:")
    print(f"B->DF: {d_mat[id, 1, 0]:.4f} (p={corrected_p[0]:0.3f}" + ("*)" if reject_null[0] else ")"))
    print(f"B->P: {d_mat[id, 2, 0]:.4f} (p={corrected_p[1]:0.3f}" + ("*)" if reject_null[1] else ")"))
    print(f"B->(P + DF): {d_mat[id, 3, 0]:.4f} (p={corrected_p[2]:0.3f}" + ("*)" if reject_null[2] else ")"))

    #get the second order comparisons.
    comps = s_mat[id, (2, 3, 3), (1, 1, 2)]
    reject_null, corrected_p, _, _ = multipletests(comps, alpha=0.05)

    # what comparisons do we want to see
    print("Second order comparisons to previous methods:")
    print(f"DF->P: {d_mat[id, 2, 1]:.4f} (p={corrected_p[0]:0.3f}" + ("*)" if reject_null[0] else ")"))
    print(f"DF->(P + DF): {d_mat[id, 3, 1]:.4f} (p={corrected_p[1]:0.3f}" + ("*)" if reject_null[1] else ")"))
    print(f"P->(P + DF): {d_mat[id, 3, 2]:.4f} (p={corrected_p[2]:0.3f}" + ("*)" if reject_null[2] else ")"))


    # Other comparison methodology
    print("Comparison of bootsrtapped v full")
    print(f"P->FP: {d_mat[id, 4, 2]:.4f} (p={s_mat[id, 4, 2]:0.3f})" + ("*)" if s_mat[id, 4, 2] < 0.05 else ")"))


    #get the second order comparisons.
    comps = s_mat[id, (2, 3, 3), (1, 1, 2)]

    print(" ")
    print(" ")
    # how much bettfor s in s_mat:

#
#     #for the first row of results
#
#     print()
#
#     # design the tests that we expect to se
#
#     # compare everything to the baseline with a t/test
#     # do a paired comparison for everything.
#     # can