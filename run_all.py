"""Launches all experiments"""
from argparse import ArgumentParser
import os
import subprocess


def main():
    argp = ArgumentParser()
    argp.add_argument("base_dir", help="Base directory that stores configs. Likely 'configs'")
    argp.add_argument("--blacklist", help="Don't run experiment configs with any of these comma-separated substrings in their filenames")
    argp.add_argument("--whitelist", help="Only run experiment configs with these comma-separated substrings in their filenames")
    args = argp.parse_args()

    blacklist = args.blacklist.split(",") if args.blacklist else None
    whitelist = args.whitelist.split(",") if args.whitelist else None

    for dir_name, sub_dirs, file_list in os.walk(args.base_dir):
        for config in file_list:
            path = os.path.join(dir_name, config)
            if blacklist and any([item in path for item in blacklist]):
                continue

            if whitelist and not any([item in path for item in whitelist]):
                continue

            cmd = f"python run.py {path}"

            print(f"> {cmd}")
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
