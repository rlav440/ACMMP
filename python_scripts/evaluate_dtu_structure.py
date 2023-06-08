import logging
from pathlib import Path

import tqdm

from select_dtu_cams import ReconParams
import subprocess
import shutil
import numpy as np
import pyvista as pv
import cv2
from natsort import natsorted
import glob

from abiStereoRaySampler import probaliblity_volume

from pyCamera import Camera


def load_camera(file_name, res=(1600, 1200)) -> Camera:
    ext = np.loadtxt(file_name, skiprows=1, max_rows=4)
    int = np.loadtxt(file_name, skiprows=7, max_rows=3)
    return Camera(extrinsic=ext, intrinsic=int, res=res)

r_param = ReconParams(mindist=300, maxdist=800, maxangle=120)
# for dtu
loc_input = Path('../recons/dtu_eval')
acmmp_loc = Path("../cmake-build-debug/ACMMP")

to_run = natsorted(list(loc_input.iterdir()))



for outloc in tqdm.tqdm(to_run):
    logging.info(f"Working on {outloc}")

    if not (outloc/'pair.txt').exists():
        logging.error("Download error: skipping.")
        continue

    try:
        if not (outloc/"ACMMP_no_prior.ply").exists():
            subprocess.run([acmmp_loc, str(outloc) + "/"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                           )
            shutil.copy(outloc / "ACMMP/ACMMP_model.ply", outloc / "ACMMP_no_prior.ply")

        if not (outloc/"ACMMP_x2.ply").exists():
            # copy this data to a new location
            subprocess.run([
                acmmp_loc, str(outloc) + "/",
                "--output_dir", "ACMMP2/",
                "--multi_fusion", "ACMMP/", "--force_fusion"
            ],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
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
            cam_loc = natsorted(glob.glob(str(outloc/"cams") + "/*_cam.txt"))
            cam_subset = [load_camera(cam_f) for cam_f in cam_loc]

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
                "-p", "--multi_fusion",
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )

            shutil.move(outloc / 'ACMMP_PRIOR/ACMMP_prior_model.ply', outloc / 'acmmp_boost_1.ply')
            # run the probability volume

        try:
            shutil.rmtree(outloc/"ACMMP")
            shutil.rmtree(outloc/"ACMMP2")
            shutil.rmtree(outloc/"ACMMP_PRIOR")
        except:
            pass

    except Exception as e:

        logging.error(f"failed on directory {outloc}")
        logging.error("Given exception: " + str(e))