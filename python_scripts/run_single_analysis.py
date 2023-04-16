import logging
from pathlib import Path
from natsort import natsorted
import pyvista as pv
import glob
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import subprocess

from abiStereoRaySampler import probaliblity_volume
from pyCamera import Camera

from select_dtu_cams import setup_from_source, ReconParams


# load all the cameras from the calibration file.
def load_camera(file_name, res=(1600, 1200)) -> Camera:
    ext = np.loadtxt(file_name, skiprows=1, max_rows=4)
    int = np.loadtxt(file_name, skiprows=7, max_rows=3)
    return Camera(extrinsic=ext, intrinsic=int, res=res)



# for a range of reconstruction samples:
r_param = ReconParams(mindist=300, maxdist=800, maxangle=120)
# # for dtu
# loc_input = Path('../recons/dtu')
# loc_output = Path('../recons/programmatic')
acmmp_loc = Path("../cmake-build-debug/ACMMP")
# good_cams = Path("../recons/sketch/dtu_scan6/cams")

target_dir = Path("../recons/sketch/cam_five_dense")
outloc = target_dir

# run the reconstruction
subprocess.run(
    [acmmp_loc, str(outloc) + "/", "-f", "0.05"],
    # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
# grab the output file
shutil.copy(outloc/"ACMMP/ACMMP_model.ply", outloc / "ACMMP_no_prior.ply")
ptcloud = pv.read(outloc/"ACMMP_no_prior.ply")

logging.info("building output volume")
sampler = probaliblity_volume(
    points=np.array(ptcloud.points),
    gaussian_dist=10,
    caching=False,
)

logging.info("Generating per camera sample estimates")
cam_loc = natsorted(glob.glob(str(target_dir/"cams/*_cam.txt")))
cam_targets = [load_camera(cam_f) for cam_f in cam_loc]

for ind, cam in enumerate(cam_targets):
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
    # shutil.rmtree(outloc/"ACMMP") # only needed for repeated runs


# run the processing script to generate priors
subprocess.run(
    [acmmp_loc, str(outloc) + '/', '-p', '-f', "0.3"],
    # stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
)

shutil.move(outloc / 'ACMMP/ACMMP_model.ply', outloc / 'acmmp_boost_1.ply')
# run the probability volume
