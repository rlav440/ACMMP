from scipy.io import loadmat
from matplotlib import pyplot as plt
from pathlib import Path
import cv2
import numpy as np

loc = Path("../bin/20 MP Stereo Pair/")
input_loc = Path("../recons/alex_recon_test")

# @dataclass
class ReconParams:
    def __init__(
            self, mindist=0.1, maxdist=0.8, steps=192, minangle=5, maxangle=45,
            max_n_view=9
    ):
        self.mindist = mindist
        self.maxdist = maxdist
        self.steps = steps
        self.minangle = minangle
        self.maxangle = maxangle
        self.max_n_view = max_n_view


def write_cam_file(f_loc, extrinsic, intrinsic, r_param: ReconParams):
    with open(f_loc, "w") as f:
        # split the file
        f.write('extrinsic' + '\n')
        for row in extrinsic:
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}" + '\n')
        # write the input
        f.write('\n')
        f.write("intrinsic" + '\n')
        for row in intrinsic:
            f.write(f"{row[0]} {row[1]} {row[2]}" + '\n')
        # write the intrinsic matrix
        f.write('\n')
        # write the relevant params for the camera.
        f.write(
            f"{r_param.mindist} {(r_param.maxdist - r_param.mindist) / r_param.steps} {r_param.steps} {r_param.maxdist}" + '\n')


intrinsicMatrix1 = 1.0e+03 * np.array([[2.6098, 0, 2.7333], [0, 2.6089, 1.8121], [0, 0, 0.0010]])
distortionCoefficients1 = np.array([-0.1644, 0.0675, 0, 0, 0])[:, None]
intrinsicMatrix2 = 1.0e+03 * np.array([[2.6211, 0, 2.6952], [0, 2.6207, 1.8224], [0, 0, 0.0010]])
distortionCoefficients2 = np.array([-0.1643, 0.0662, 0, 0, 0])[:, None]
rotationOfCamera2 = np.array([[1.0000, 0.0028, 0.0015], [-0.0028, 1.0000, 0.0025], [-0.0015, -0.0025, 1.0000]])
translationOfCamera2 = np.array([-59.7113, 0.0266, 0.2548])

cam0_ext = np.eye(4)
cam1_ext = np.block([
    [rotationOfCamera2, translationOfCamera2[:, None]], [0,0,0,1]
])

# handle creating the input for ACMMP

#load the images
im0 = cv2.imread(str(loc/'cam0_2.png'))
im1 = cv2.imread(str(loc/'cam1_2.png'))

# undistort the two images, after checking the params are correct
new_im1 = cv2.undistort(im1, intrinsicMatrix2, distortionCoefficients2, None, intrinsicMatrix2)
new_im0 = cv2.undistort(im0, intrinsicMatrix1, distortionCoefficients1, None, intrinsicMatrix1)

input_loc.mkdir(parents=True, exist_ok=True)
cams = input_loc/'cams'
cams.mkdir(parents=True, exist_ok=True)
images = input_loc/'images'
images.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(images/f"{0:08}.jpg"), new_im0)
cv2.imwrite(str(images/f"{1:08}.jpg"), new_im1)

r_param = ReconParams(mindist=500, maxdist=3000)
write_cam_file(cams/f"{0:08}_cam.txt", cam0_ext, intrinsicMatrix1, r_param)
write_cam_file(cams/f"{1:08}_cam.txt", cam1_ext, intrinsicMatrix2, r_param)

