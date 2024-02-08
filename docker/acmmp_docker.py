import subprocess
import argparse


################
### COMMANDS ###
################

'''
python3 acmmp_docker.py \
--acmmp_output_dir /mnt/nvme_fast/acmmp_pipeline_test/acmmp_hand_robin_20230628_182542/acmmp_output

python3 acmmp_docker.py \
--num_consistent_thresh 3 \
--acmmp_output_dir /mnt/nvme_fast/acmmp_pipeline_test/acmmp_hand_robin_20230628_182542/acmmp_output
'''


########
# TODO 
# 1 New names for input args
# 2 Arbitrary file extensions in ACMMP 
# 3 Arbitrary secondary cameras for texture mapping during fusion 
########
print("TODO\n")
print("New names for input args")



# --------------------------------------------------
# STEP 0 - Setup global parameters to pass through 
# --------------------------------------------------

# Create an argument parser
parser = argparse.ArgumentParser()

# Add command-line options
parser.add_argument('--help_acmmp', help='produce help infor for ACMMP program')
parser.add_argument('--prior', help='runs the reconstruction from a provided prior')
parser.add_argument('--fuse_thresh', type=float, help='arg | <0.30 | Sets the average inverse score threshold for fusion ')
parser.add_argument('--dense_folder', type=str, help='arg | The input folder for reconstruction')
parser.add_argument('--multi_fusion', type=str, help='arg | default -> /ACCP | Use information from a previous reconstruction during fusion of invididual camera')
parser.add_argument('--force_fusion', help='forces multi fusion, without prior')
parser.add_argument('--output_dir', help='arg | default -> /ACMMP | Output working directory name')
parser.add_argument('--num_consistent_thresh', type=str, help='arg | ~3 | Number of points that must be consistent to be fused into the final output pointcloud. (Reduces noisy output)')
parser.add_argument('--single_match_penalty', type=str, help='arg | An increase to the consistency threshold for matched hypotheses that only matched over a single set')
parser.add_argument('--acmmp_output_dir', type=str, help='arg | Absolute path to acmmp_output folder')

# Parse the command-line arguments
args = parser.parse_args()

# Create a list of flags and values based on the provided command-line arguments
flags = []
if args.help_acmmp:
    flags.extend(['--help_acmmp', args.help_acmmp])
if args.prior:
    flags.extend(['--prior', args.prior])
if args.fuse_thresh:
    flags.extend(['--fuse_thresh', args.flag3])
if args.dense_folder:
    flags.extend(['--dense_folder', args.dense_folder])
if args.multi_fusion:
    flags.extend(['--multi_fusion', args.multi_fusion])
if args.force_fusion:
    flags.extend(['--force_fusion', args.force_fusion])
if args.output_dir:
    flags.extend(['--output_dir', args.output_dir])
if args.num_consistent_thresh:
    flags.extend(['--num_consistent_thresh', args.num_consistent_thresh])
if args.single_match_penalty:
    flags.extend(['--single_match_penalty', args.single_match_penalty])

# extract working directory string
acmmp_output_dir = args.acmmp_output_dir

# DEBUG
# print(f"DEBUG -> 'acmmp_output' directory:",  acmmp_output_dir)
# print(f"DEBUG -> ACMMP Flags: {flags}")






# --------------------
# STEP 5 - Run ACMMP #
# --------------------

'''
-h [ --help ]                  produce help message
-p [ --prior ]                 runs the reconstruction from a provided prior
-f [ --fuse_thresh ] arg (float)     Sets the average inverse score threshold for 
                                fusion
--dense_folder arg (str)             The input folder for reconstruction
--multi_fusion [=arg(=/ACMMP)] (str) Use information from a previous reconstruction
                                during fusion of invididual camera 
                                reconstructions
--force_fusion                 forces multi fusion, without prior
--output_dir [=arg(=/ACCMP)] (str) Output working directory name
--num_consistent_thresh arg (int)   Number of points that must be consistent to be
                                fused into the final output pointcloud.
--single_match_penalty arg (int)   An increase to the consistency threshold for 
                                matched hypotheses that only matched over a 
                                single set
                                
  
## ADDING ADDITIONAL FLAGS 

import argparse
import subprocess

# Define the command-line interface
parser = argparse.ArgumentParser()
parser.add_argument('--flag1', help='Flag 1 description')
parser.add_argument('--flag2', help='Flag 2 description')
parser.add_argument('--flag3', help='Flag 3 description')
parser.add_argument('file_path', help='File path argument')

# Parse the command-line arguments
args = parser.parse_args()

# Create a list of flags and values based on the provided command-line arguments
flags = []
if args.flag1:
    flags.extend(['--flag1', args.flag1])
if args.flag2:
    flags.extend(['--flag2', args.flag2])
if args.flag3:
    flags.extend(['--flag3', args.flag3])

# Add more flags and values as needed based on your requirements

# Run the subprocess with the specified flags, values, and file path
subprocess.run(['your_command', *flags, args.file_path])
'''

# DEBUG
run_acmmp = True

if run_acmmp == True:
    # path to acmmp binary
    binary_path = "/ACMMP_docker_build/ACMMP"
    # Run the subprocess with the specified flags, values, and file path
    subprocess.run([binary_path, *flags], check=True)








