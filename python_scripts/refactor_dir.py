from pathlib import Path




def restring_dir(dur: Path):
    for floc in dur.glob("*"):
        with open(floc, 'r') as f:
            lines = f.readlines()
            lines[-1] = "300 2.6041666666666665 192 800"
        with open(floc, 'w') as f:
            f.writelines(lines)




if __name__ == '__main__':
    targ_dir = Path("../recons/sketch/dtu_scan9/cams")
    restring_dir(targ_dir)
