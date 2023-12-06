import os
import sys

def save_script_args():
    CMD = f"python3 {' '.join(sys.argv)}\n"
    outpath = 'CMDs/{}'.format(sys.argv[0].replace('py', 'cmds'))
    outdir = os.path.dirname(os.path.abspath(outpath))
    if not os.path.exists(outdir):
        os.makedirs(outdir)    
    with open(outpath, 'a+') as f:
        f.write(CMD)

def check_output_path(output_path):
    if os.path.exists(output_path):
        exit(f"{output_path} already exists!")

    outdir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

def check_output_dir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")