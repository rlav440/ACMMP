from pathlib import Path
import shutil


floc = Path("../recons/programmatic_run0")

for recon in floc.iterdir():
    shutil.rmtree(recon/'ACMMP')