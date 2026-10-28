"""
shoot.py - run on any computer that can reach the Jetson (macOS / Windows / Linux, Python 3 + OpenSSH)

One command: the Jetson tells the Pi to make the GoPro take N photos, pulls them, predicts them on its
GPU and makes ORIGINAL | PREDICTION comparisons (process/capture_predict.py) -> everything is moved to
this computer, a summary is printed and the folder of comparisons opens.
Nothing is kept on the Pi or the Jetson. Nothing is processed on this computer.
Setup: zip extracted in the Jetson's home (~/pool_detect), gopro_capture.py in the Pi's home,
and an SSH key from the Jetson to the Pi (on the Jetson: ssh-copy-id <user-pi>@<ip-pi>).

Usage:
    python3 shoot.py --install --jetson user@JETSON_IP       # once: copy to Jetson + Pi, build, self-test
    python3 shoot.py 10 --jetson user@JETSON_IP              # take 10 photos
    python3 shoot.py 10 --jetson user@ip --pi pi@192.168.2.2 --session test1
    python3 shoot.py 20 --jetson user@ip --interval 5              # one photo every 5 s
    python3 shoot.py --jetson user@ip --only-predict test1         # photos already on the Jetson, no capture
Result in ./image_gopro/<session>/:
    raw/        GoPro photos untouched (full size, no boxes), e.g. for training later
    compare/    compare_<name>.jpg, one ORIGINAL | PREDICTION image per photo
    predicted/  pred_<name>.jpg with boxes drawn
    results.jsonl
If no SSH key from this computer to the Jetson is set up, ssh asks for the Jetson password.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ssh_opts():
    opts = ["-o", "ConnectTimeout=10"]
    if os.name != "nt":  # reuse one connection so a password is asked only once (not supported on Windows)
        opts += ["-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/aiot-%r@%h-%p", "-o", "ControlPersist=120"]
    return opts


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def open_file(path):
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)])
    except OSError:
        pass


def summary(results):
    """Print what the Jetson found in each photo (from results.jsonl)."""
    if not results.exists():
        return
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "counts" in r]
    names = list(rows[0]["counts"]) if rows else []
    total = {n: 0 for n in names}
    print("\n{:<32}".format("file") + "".join("{:>9}".format(n) for n in names) + "{:>8}{:>14}".format("total", "inference ms"))
    for r in rows:
        for n in names:
            total[n] += r["counts"][n]
        print("{:<32}".format(r["file"][:31]) + "".join("{:>9}".format(r["counts"][n]) for n in names)
              + "{:>8}{:>14}".format(sum(r["counts"].values()), r["inference_ms"]))
    print("{:<32}".format("ALL ({} photos)".format(len(rows))) + "".join("{:>9}".format(total[n]) for n in names)
          + "{:>8}".format(sum(total.values())))


def install(args, ssh):
    """Copy the Jetson part and the Pi script into place, build the engine if needed and self-test."""
    root = HERE.parent
    print("== 1/3 copy models/ process/ test/ to the Jetson ~/" + args.remote_dir, flush=True)
    if run(ssh + [args.jetson, "mkdir -p {}/images".format(args.remote_dir)]) or run(
            ["scp", "-r"] + ssh_opts() + [str(root / d) for d in ("models", "process", "test")]
            + ["{}:{}/".format(args.jetson, args.remote_dir)]):
        sys.exit("could not copy to the Jetson")
    print("== 2/3 self-test on the Jetson (first time also builds the engine: 5-15 min)", flush=True)
    test_ok = run(ssh + [args.jetson, "cd {}/process && python3 -u selftest.py".format(args.remote_dir)]) == 0
    if args.no_pi:
        print("== 3/3 Pi skipped (--no-pi)")
    else:
        print("== 3/3 copy gopro_capture.py to the Pi " + args.pi, flush=True)
        if run(["scp"] + ssh_opts() + ["-o", "ProxyJump=" + args.jetson, str(root / "capture/gopro_capture.py"),
                                       "{}:{}".format(args.pi, args.pi_script)]):
            print("could not reach the Pi; the Jetson part still works (--only-predict)")
    sys.exit(0 if test_ok else "SELF-TEST FAILED on the Jetson (see above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("count", type=int, nargs="?", default=0, help="number of photos to take")
    ap.add_argument("--jetson", required=True, help="user@ip of the Jetson as seen from this computer")
    ap.add_argument("--pi", default="pi@192.168.2.2", help="user@ip of the Pi as seen from the Jetson")
    ap.add_argument("--pi-script", default="gopro_capture.py", help="gopro_capture.py on the Pi (relative to its home)")
    ap.add_argument("--remote-dir", default="pool_detect", help="folder on the Jetson holding process/ models/ images/")
    ap.add_argument("--session", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--interval", type=float, default=0, help="seconds between shots")
    ap.add_argument("--legacy", action="store_true", help="GoPro HERO5-HERO8")
    ap.add_argument("--camera", help="last digits of the GoPro name, e.g. 8488")
    ap.add_argument("--model", default="model_X3", help="model on the Jetson: model_X3 (default)")
    ap.add_argument("--only-predict", metavar="SESSION", help="skip capture; predict a session already on the Jetson")
    ap.add_argument("--out", type=Path, default=HERE / "image_gopro")
    ap.add_argument("--install", action="store_true", help="copy everything to the Jetson and Pi, build, self-test")
    ap.add_argument("--no-pi", action="store_true", help="with --install: Jetson only")
    args = ap.parse_args()
    if args.install:
        install(args, ["ssh"] + ssh_opts())
    if args.only_predict:
        args.session = args.only_predict
    elif args.count < 1:
        ap.error("give the number of photos, or --only-predict SESSION")

    ssh = ["ssh"] + ssh_opts()
    remote_session = "{}/images/{}".format(args.remote_dir, args.session)
    if args.only_predict:
        job = "python3 -u predict_folder.py {} --model {}".format(args.session, args.model)
    else:
        job = "python3 -u capture_predict.py {} --pi {} --pi-script {} --session {} --interval {}{}{}".format(
            args.count, args.pi, args.pi_script, args.session, args.interval, " --legacy" if args.legacy else "",
            " --camera " + args.camera if args.camera else "") + " --model " + args.model
    print("== 1/2 on the Jetson: {}".format("predict" if args.only_predict else "capture via Pi + predict"), flush=True)
    job_ok = run(ssh + ["-t", args.jetson, "cd {}/process && {}".format(args.remote_dir, job)]) == 0
    if not job_ok and run(ssh + [args.jetson, "test -d " + remote_session]):
        sys.exit("nothing to copy back (see the messages above)")

    print("== 2/2 move results to " + str(args.out / args.session), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if run(["scp", "-r"] + ssh_opts() + ["{}:{}".format(args.jetson, remote_session), str(args.out)]):
        sys.exit("copy failed; results are still on the Jetson in " + remote_session)
    run(ssh + [args.jetson, "rm -rf " + remote_session])  # copied, so nothing stays on the Jetson

    local = args.out / args.session
    summary(local / "results.jsonl")
    print("raw: {}  compare: {}  -> {}".format(
        len(list((local / "raw").glob("*"))), len(list((local / "compare").glob("*"))), local))
    if (local / "compare").is_dir():
        open_file(local / "compare")
    if not job_ok:
        sys.exit("the Jetson reported errors (see above); showing what was there")


if __name__ == "__main__":
    main()
