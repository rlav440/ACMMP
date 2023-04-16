import subprocess
from pathlib import Path
import numpy as np
from matlab.engine import start_matlab
from multiprocessing.pool import Pool

import pyvista as pv
from natsort import natsorted
from tqdm import tqdm
import  re
import io

from abiStereoRaySampler.DTU_eval import dtu_eval

eng = start_matlab()
eng.addpath('/home/rlav440/CLionProjects/gipuma/scripts/data/dtu/SampleSet/Matlab evaluation code')
datapath = "/home/rlav440/CLionProjects/gipuma/scripts/data/dtu/SampleSet/MVS Data"

working_loc = Path("../recons/programmatic_run1")
base_loc = Path("/home/rlav440/CLionProjects/gipuma/scripts/data/dtu/SampleSet/MVS Data/Points/stl")
dst = 0.2
#for every scan

# do the comparison on both the mvs and the non mavs data

floc = Path("../recons/programmatic_run1")
pv.set_plot_theme("Document")
tests = natsorted(list(floc.iterdir()))

# eng = start_matlab()
# eng.addpath('/home/rlav440/CLionProjects/gipuma/scripts/data/dtu/SampleSet/Matlab evaluation code')

def analyse_path(recon: Path) -> None:

    f_name = recon.parts[-1].split("_")[0]
    camset_num = int(re.findall(r"\d+", f_name)[0])

    ref_loc = base_loc/f"stl{camset_num:03}_total.ply"

    obj_loc0 = str(recon/"ACMMP_no_prior.ply")
    outname0 = str(recon/"acmmp_no_prior.txt")

    # eng.run_matlab_analysis(camset_num, datapath, obj_loc0, dst, outname0, nargout=0, stdout=io.StringIO())
    dtu_eval(
        pv.read(ref_loc).points,
        pv.read(obj_loc0).points,
        outname0,
    )


    obj_loc0 = str(recon/"ACMMP_x2.ply")
    outname0 = str(recon/"acmmp_x2.txt")
    dtu_eval(
        pv.read(ref_loc).points,
        pv.read(obj_loc0).points,
        outname0,
    )
    # eng.run_matlab_analysis(camset_num, datapath, obj_loc0, dst, outname0, nargout=0, stdout=io.StringIO())

    obj_loc1 = str(recon/"acmmp_boost_1.ply")
    outname1 = str(recon/"acmmp_boosted.txt")
    dtu_eval(
        pv.read(ref_loc).points,
        pv.read(obj_loc1).points,
        outname1,
    )
    # eng.run_matlab_analysis(camset_num, datapath, obj_loc1, dst, outname1, nargout=0, stdout=io.StringIO())



for t in tqdm(tests):
    analyse_path(t)

# eng.run_matlab_analysis(camera_set, datapath, obj_loc, dst, outname, nargout=0)
# eng.quit()




