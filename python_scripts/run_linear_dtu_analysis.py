import glob
import numpy as np
import shutil
import subprocess
from pathlib import Path
from natsort import natsorted
import pyvista as pv
import cv2
from tqdm import tqdm

from abiStereoRaySampler import probaliblity_volume
from pyCamera import Camera

from select_dtu_cams import setup_from_source, ReconParams


# load all the cameras from the calibration file.
def load_camera(file_name, res=(1600, 1200)) -> Camera:
    ext = np.loadtxt(file_name, skiprows=1, max_rows=4)
    int = np.loadtxt(file_name, skiprows=7, max_rows=3)
    return Camera(extrinsic=ext, intrinsic=int, res=res)


cam_loc = natsorted(glob.glob("../recons/sketch/dtu_scan6/cams/*_cam.txt"))
cam_models = [load_camera(cam_f) for cam_f in cam_loc]

cam_sets = {
    2: [19, 27],
    3: [19, 23, 27],
    5: [19, 21, 23, 25, 27],
    9: [19, 20, 21, 22, 23, 24, 25, 26, 27],
}

# for a range of reconstruction samples:
r_param = ReconParams(mindist=300, maxdist=800, maxangle=120)
# for dtu
loc_input = Path('../recons/dtu')
loc_output = Path('../recons/programmatic_run1')
acmmp_loc = Path("../cmake-build-debug/ACMMP")
good_cams = Path("../recons/sketch/dtu_scan6/cams")

to_run = list(loc_input.iterdir())

# to_run = [
#     Path('../recons/dtu/scan77'),
#     Path("../recons/dtu/scan110")
# ]



# take a subset of these reults
for target_dir in tqdm(to_run):

    # run make
    for n_cam, cam_targets in cam_sets.items():
        outloc = loc_output / (target_dir.parts[-1] + f"_{n_cam}_cam")

        shutil.rmtree(target_dir / "cams")
        shutil.copytree(good_cams, target_dir / "cams")
        setup_from_source(cam_targets, target_dir, outloc, r_param)
        # run the reconstruction
        # standard data standard reconstruction
        if not (outloc/"ACMMP_no_prior.ply").exists():
            subprocess.run([acmmp_loc, str(outloc) + "/"],
                           # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                           )
            shutil.copy(outloc / "ACMMP/ACMMP_model.ply", outloc / "ACMMP_no_prior.ply")

        if not (outloc/"ACMMP_x2.ply").exists():
            # copy this data to a new location
            subprocess.run([
                acmmp_loc, str(outloc) + "/",
                "--output_dir", "ACMMP2/",
                "--multi_fusion", "ACMMP/", "--force_fusion"
                ],
                # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            shutil.copy(outloc / "ACMMP2/ACMMP_prior_model.ply", outloc / "ACMMP_x2.ply")

        # grab the output file
        if not (outloc/"acmmp_boost_1.ply").exists():
            ptcloud = pv.read(outloc / "ACMMP_no_prior.ply")

            sampler = probaliblity_volume(
                points=np.array(ptcloud.points),
                gaussian_dist=10,
                caching=False,
            )
            cam_subset = [cam_models[ind] for ind in cam_targets]
            for ind, cam in enumerate(cam_subset):
                depths, normals = sampler.sample(
                    cam,
                    min_dist=r_param.mindist,
                    max_dist=r_param.maxdist,
                )
                prior_loc = outloc / "priors"
                d_loc = prior_loc / 'depths'
                d_loc.mkdir(parents=True, exist_ok=True)
                n_loc = prior_loc / 'normals'
                n_loc.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(prior_loc / f'depths/{ind:08}.png'), depths)
                cv2.imwrite(str(prior_loc / f'normals/{ind:08}.png'), normals)



            # run the processing script to generate priors
            subprocess.run([
                acmmp_loc, str(outloc) + '/',
                "-p", "--multi_fusion"
            ],
                # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )

            shutil.move(outloc / 'ACMMP_PRIOR/ACMMP_prior_model.ply', outloc / 'acmmp_boost_1.ply')
        # run the probability volume
        shutil.rmtree(outloc/"ACMMP")
        shutil.rmtree(outloc/"ACMMP2")
        shutil.rmtree(outloc/"ACMMP_PRIOR")
