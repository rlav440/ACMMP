import logging
import re
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
loc_clouds = Path("/home/rlav440/CLionProjects/gipuma/scripts/data/dtu/SampleSet/MVS Data/Points/stl/")

to_run = natsorted(list(loc_input.iterdir()))


#what else do we want to do:



for outloc in tqdm.tqdm(to_run):
    logging.info(f"Working on {outloc}")

    if not (outloc/'pair.txt').exists():
        logging.error("Download error: skipping.")
        continue

    try:
        # run with the default prior and only do a reconstruction with hust those results
        if not (outloc/"acmmp_boost_single.ply").exists():
            ptcloud = pv.read(outloc / "ACMMP_no_prior.ply")

            # run the processing script to generate priors

            sampler = probaliblity_volume(
                points=np.array(ptcloud.points),
                gaussian_dist=5,
                caching=False,
            )

            # sampler.draw_volume()
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

            subprocess.run([
                acmmp_loc, str(outloc) + '/',
                "-p",
                           ],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            shutil.move(outloc/'ACMMP_PRIOR/ACMMP_model.ply', outloc/'acmmp_boost_single.ply')
            # run the probability volume

        # run the reconstruction with a fully known prior

        if not (outloc/"ACMMP_full_prior.ply").exists():

            #get the name of the scan
            scan_file = outloc.parts[-1].split("_")[0]
            cloud_n = int(re.findall(r'\d+', str(scan_file))[0])


            # grab the prior
            ptcloud = pv.read(str(loc_clouds/f'stl{cloud_n:03}_total.ply'))
            # down sample by a big factor
            points = ptcloud.points
            inds = np.random.choice(np.arange(points.shape[0]), points.shape[0]//100)
            # pv.PolyData(points[inds]).plot()
            sampler = probaliblity_volume(
                points=np.array(points[inds]),
                gaussian_dist=5,
                caching=False,
            )

            # sampler.draw_volume()
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


            subprocess.run([
                acmmp_loc, str(outloc) + '/',
                "-p",
                "--output_dir", "ACMMP_full_prior"
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            shutil.move(outloc /"ACMMP_full_prior/ACMMP_model.ply", outloc / "ACMMP_full_prior.ply")


        try:
            shutil.rmtree(outloc/"ACMMP_full_prior")
            shutil.rmtree(outloc/"ACMMP_PRIOR")
        except:
            pass

    except Exception as e:
        logging.error(f"failed on directory {outloc}")
        logging.error("Given exception: " + str(e))
