import tqdm
from pathlib import Path
from select_dtu_cams import ReconParams
import subprocess
import shutil
import numpy as np
import pyvista as pv
import cv2
from natsort import natsorted
import glob

from abiStereoRaySampler import probaliblity_volume

from pyCamera import Camera, CameraSet


def load_camera(file_name, res=(1600, 1200)) -> Camera:
    ext = np.loadtxt(file_name, skiprows=1, max_rows=4)
    int = np.loadtxt(file_name, skiprows=7, max_rows=3)
    return Camera(extrinsic=ext, intrinsic=int, res=res)

r_param = ReconParams(mindist=300, maxdist=800, maxangle=120)
# for dtu
cam_loc = Path("../recons/dtu/scan1/cams")
scams = natsorted([str(f) for f in cam_loc.iterdir()])[:49]
cam_dict = {str(id):load_camera(cam) for id, cam in enumerate(scams)}


ptcloud = pv.read("~/CLionProjects/gipuma/scripts/data/dtu/SampleSet/MVS Data/Points/stl/stl033_total.ply")

cams = CameraSet(camera_dict=cam_dict)
# cams.plot(additional_mesh=ptcloud)

new_cam = cams.make_subset(list(range(19)) + list(range(28, 49)))

good_cams = cams.make_subset(list(range(19, 28)))
good_cams.plot(additional_mesh=ptcloud)

