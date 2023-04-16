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
def load_camera(file_name, res=(1600,1200)) -> Camera:
    ext = np.loadtxt(file_name, skiprows=1, max_rows=4)
    int = np.loadtxt(file_name, skiprows=7, max_rows=3)
    return Camera(extrinsic=ext, intrinsic=int, res=res)


cam_loc = natsorted(glob.glob("../recons/sketch/dtu_scan6/cams/*_cam.txt"))
cam_models = [load_camera(cam_f) for cam_f in cam_loc]

cam_sets = {
    2: [38, 48],
    3: [38, 8, 48],
    4: [38, 8, 48, 43],
    5: [13, 17, 38, 43, 48],
    6: [8, 22, 26, 38, 43, 48],
    7: [0, 4, 25, 21, 38, 43, 48],
    8: [0, 4, 8, 21, 26, 38, 43, 48],
    9: [0, 4, 19, 23, 27, 38, 42, 45, 48],
    10: [0, 4, 19, 22, 25, 27, 38, 42, 45, 48]
}

# for a range of reconstruction samples:
r_param = ReconParams(mindist=300, maxdist=800, maxangle=120)
# for dtu
loc_input = Path('../recons/dtu')
loc_output = Path('../recons/programmatic_run0')
acmmp_loc = Path("../cmake-build-debug/ACMMP")
good_cams = Path("../recons/sketch/dtu_scan6/cams")

to_run = list(loc_input.iterdir())
for target_dir in tqdm(to_run):
    # run make
    for n_cam, cam_targets in cam_sets.items():
        outloc = loc_output / (target_dir.parts[-1] + f"_{n_cam}_cam")

        shutil.rmtree(target_dir / "cams")
        shutil.copytree(good_cams, target_dir / "cams")
        setup_from_source(cam_targets, target_dir, outloc, r_param)
        # run the reconstruction
        subprocess.run([acmmp_loc, str(outloc) + "/"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # grab the output file
        shutil.copy(outloc / "ACMMP/ACMMP_model.ply", outloc / "ACMMP_no_prior.ply")
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

            shutil.rmtree(outloc/"ACMMP")

        # run the processing script to generate priors
        subprocess.run([acmmp_loc, str(outloc) + '/'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        shutil.move(outloc / 'ACMMP/ACMMP_model.ply', outloc / 'acmmp_boost_1.ply')
        # run the probability volume
