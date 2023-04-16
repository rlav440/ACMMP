from dataclasses import dataclass
from io import TextIOWrapper
import numpy as np
from pathlib import Path
from natsort import natsorted
import shutil

# this script:
# selects the cameras to be used for a dtu camera reconstruction
# loads the selected cameras
# builds pair listings.
# outputs the results to a new folder

@dataclass
class ReconParams:
    def __init__(
            self, mindist=0.1, maxdist=0.8, steps=192, minangle=3, maxangle=45,
            max_n_view=9
    ):
        self.mindist = mindist
        self.maxdist = maxdist
        self.steps = steps
        self.minangle = minangle
        self.maxangle = maxangle
        self.max_n_view = max_n_view


def write_pair_file(f: TextIOWrapper, pair_list):
    f.write(f"{int(len(pair_list))}" + '\n')
    for idi, list_vals in enumerate(pair_list):
        f.write(f"{idi}" + '\n')
        line_string = f"{len(list_vals)} "
        line_string += " ".join([f"{cam_id} 1" for cam_id in list_vals])
        f.write(line_string + '\n')
    return


def calc_pairs(c_vec, r_param: ReconParams, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    c_vec /= np.linalg.norm(c_vec, axis=1, keepdims=True)
    t = c_vec[None, ...] * c_vec[:, None]
    ang = np.arccos(np.sum(t, axis=-1)) * 180 / np.pi
    mask = np.logical_and(
        ang > r_param.minangle, ang < r_param.maxangle
    )
    returned_pairs = []
    for masklet in mask:
        valid_points = np.where(masklet)[0]
        if len(valid_points) < r_param.max_n_view:
            returned_pairs.append(valid_points)
        else:
            returned_pairs.append(
                rng.choice(valid_points, r_param.max_n_view)
            )
    return returned_pairs

def read_ext(f: TextIOWrapper):
    return np.loadtxt(f, skiprows=1, max_rows=4)

def get_v_vec(ext):
    return ext[:3,:3] @ np.array([0,0,1])

def setup_from_source(cams, src: Path, dst:Path, recon_params: ReconParams):

    cam_locs = natsorted([str(f) for f in (src/"cams").glob("*")])
    cam_vv = np.array(
        [get_v_vec(read_ext(cam_locs[c_n])) for c_n in cams]
    )
    rng = np.random.default_rng(42)
    pairs = calc_pairs(c_vec=cam_vv, r_param=recon_params, rng=rng)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    cam_dir = dst/'cams'
    cam_dir.mkdir()
    im_dir = dst/'images'
    im_dir.mkdir()

    imlist = natsorted([str(f) for f in (src/"images").glob("*")])
    camlist = natsorted([str(f) for f in (src/"cams").glob("*")])

    for idn, cam_n in enumerate(cams):
        shutil.copy(camlist[cam_n], cam_dir/f"{idn:08}_cam.txt")
        shutil.copy(imlist[cam_n], im_dir/f"{idn:08}.jpg")
    with open(dst/"pair.txt", 'w') as f:
        write_pair_file(f, pairs)


if __name__ == '__main__':
    source_loc = Path("../recons/dtu/scan62")
    output = Path("../recons/sketch/dtu62_cams_5")

    r_param = ReconParams(mindist=300, maxdist=800, maxangle=120)
    cams = [13, 17, 38, 43, 48]
    setup_from_source(cams, source_loc, output, r_param)