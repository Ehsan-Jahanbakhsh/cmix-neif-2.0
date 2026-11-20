"""Show a running comp9 encoder's processes and file progress from Linux or WSL."""
import argparse
import os
from pathlib import Path
import subprocess


def snapshot(directory):
    processes = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.joinpath("cwd").readlink() != directory:
                continue
            args = proc.joinpath("cmdline").read_bytes().split(b"\0")
            if os.path.basename(args[0]) == b"comp9":
                processes.append((proc, args))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if not processes:
        print("No comp9 encoder running in", directory)
        return

    print("Elapsed time, cumulative CPU time and memory for this run", flush=True)
    subprocess.run(["ps", "-o", "pid,etime,time,pcpu,nlwp,rss,args", "-p",
                    ",".join(proc.name for proc, _ in processes)], check=False)
    print("\nOpen files (input positions include read-ahead; percentages are per file)")
    for proc, args in processes:
        if len(args) >= 4 and args[1].lower() == b"-d":
            print("Unpacking", os.fsdecode(args[2]))
        try:
            descriptors = list((proc/"fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for fd in descriptors:
            try:
                path = fd.readlink()
                if path.parent != directory or path.name in {"ppm.page", "comp9"}:
                    continue
                size = fd.stat().st_size
                info = dict(line.split(":", 1) for line in
                            (proc/"fdinfo"/fd.name).read_text().splitlines())
                access = int(info["flags"], 8) & os.O_ACCMODE
                pos = int(info["pos"])
                if access == os.O_RDONLY and size:
                    print(f"  {path.name}   read {pos:,} / {size:,} bytes ({100*pos/size:.2f}%)")
                else:
                    print(f"  {path.name}   size {size:,} bytes")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
    snapshot(parser.parse_args().directory.resolve())
