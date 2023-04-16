from pathlib import Path
import shutil
import pyvista as pv
import vtk
import numpy as np
from natsort import natsorted
from tqdm import tqdm

test = np.array([
    [1222, 529, 289],
    [10, -29, 631],
    [-0.2, -0.4, -0.9]
])


test = [(743.6129688003306, 208.06045214797186, 530.2841481477315),
        (10.0, -29.0, 631.0),
        (-0.004358700954233047, -0.37956609265653357, -0.9251543563273338)]

def draw_with_cam(point_loc, target_dir):
    points = pv.read(point_loc)
    plotter = pv.Plotter(notebook=False,
                         off_screen=True,
                         )
    plotter.add_mesh(points, point_size=0.75, rgb=True)

    def my_cpos_callback(*args):
        print(plotter.camera_position)
        return

    # plotter.iren.add_observer(vtk.vtkCommand.EndInteractionEvent, my_cpos_callback)
    plotter.camera_position = test
    plotter.screenshot(target_dir.with_suffix(".png"))
    plotter.close()
    # plotter.show()

floc = Path("../recons/programmatic_run0")
pv.set_plot_theme("Document")
tests = natsorted(list(floc.iterdir()))
out_loc = Path("../bin/recon_evaluation")
for recon in tqdm(tests):
    f_name = recon.parts[-1].split("_")[0]
    new_dir = out_loc/f_name
    if not new_dir.exists():
        new_dir.mkdir(parents=True)

    np_dir = new_dir/"no_prior"
    if not np_dir.exists():
        np_dir.mkdir()

    p_dir = new_dir/"prior"
    if not p_dir.exists():
        p_dir.mkdir()

    point_loc = recon/'ACMMP_no_prior.ply'
    draw_with_cam(point_loc, np_dir/recon.parts[-1])

    point_loc = recon/'acmmp_boost_1.ply'
    draw_with_cam(point_loc, p_dir/recon.parts[-1])

