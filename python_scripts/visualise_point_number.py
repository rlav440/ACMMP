import pyvista as pv
from matplotlib import pyplot as plt
from pathlib import Path
import tqdm
import pandas as pd


cam_number = [2, 3, 4, 5, 6, 7, 8, 9, 10]
data = [[] for _ in cam_number]

# I have cam number -> scan_number
# should do scan_number -> cam_number

loc = Path("../recons/programmatic_run0")

for recon in tqdm.tqdm(list(loc.iterdir())):
   n = int(recon.parts[-1].split('_')[1])
   n_np = len(pv.read(recon/"ACMMP_no_prior.ply").points)
   n_p = len(pv.read(recon/"acmmp_boost_1.ply").points)
   data[n-2].append((n_p/n_np - 1) * 100)


x = pd.DataFrame(data).T
x.columns = [str(f) for f in cam_number]


fix, ax = plt.subplots(1,1)
import seaborn as sns
sns.set_theme("paper", style='ticks')
sns.boxplot(data=x, palette='vlag', color="0.1")
sns.stripplot(data=x, color="0.1")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_ylabel("Relative number of recovered points (%)")
ax.set_xlabel("Number of cameras")
ax.set_ylim([-10, 50])
ax.axhline(0, c="r")
plt.show()

fix, ax = plt.subplots(1,1)

ax.violinplot(dataset=data, positions=cam_number)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.set_ylabel("Relative number of recovered points (%)")
ax.set_xlabel("Number of cameras")

ax.set_ylim([-10, 50])
ax.axhline(0, c="r")
plt.show()
