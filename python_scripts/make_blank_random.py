import cv2
import numpy as np
from pathlib import Path

loc = Path("../bin/random")
# make a sampled image of a random type
for i in range(2):
    depths = (np.random.random((1200,1600)) * 65535).astype(np.uint16)
    normals = (np.random.random((1200,1600, 3)) * 65535).astype(np.uint16)
    cv2.imwrite(str(loc/f"depths/{i:08}.png"), depths)
    cv2.imwrite(str(loc/f"normals/{i:08}.png"), normals)

