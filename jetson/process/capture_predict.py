"""
capture_predict.py - run on the Jetson: tell the Pi to make the GoPro take N photos, pull them here,
delete them from the Pi, then predict them on this Jetson (same output as predict_folder.py).
With --send-to, the results are then moved to a computer and deleted from the Jetson.

Usage:
    python3 capture_predict.py 10 --pi pi@192.168.2.2                  # session = date-time
    python3 capture_predict.py 10 --pi pi@192.168.2.2 --session test1  # -> images/test1/
    python3 capture_predict.py 20 --pi pi@192.168.2.2 --interval 5     # one photo every 5 s
    python3 capture_predict.py 10 --pi pi@192.168.2.2 --camera 8488    # pick the GoPro named "GoPro 8488"
    python3 capture_predict.py 5 --pi pi@192.168.2.2 --send-to user@COMPUTER_IP:~/image_gopro
Needs (once): gopro_capture.py in the Pi's home, and an SSH key from this Jetson to the Pi
(ssh-keygen; ssh-copy-id pi@192.168.2.2). --send-to needs Remote Login on that computer and a key to it.
Result: images/<session>/raw/ (untouched photos), predicted/, compare/, results.jsonl
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # folder holding process/, models/, images/
SSH_OPTS = ["-o", "ConnectTimeout=10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("count", type=int, help="number of photos to take")
    ap.add_argument("--pi", default="pi@192.168.2.2", help="user@ip of the Pi")
    ap.add_argument("--pi-script", default="gopro_capture.py", help="gopro_capture.py on the Pi (relative to its home)")
    ap.add_argument("--session", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--interval", type=float, default=0, help="seconds between shots")
    ap.add_argument("--legacy", action="store_true", help="GoPro HERO5-HERO8")
    ap.add_argument("--camera", help="last digits of the GoPro name, e.g. 8488")
    ap.add_argument("--model", default="model_X3", help="model name in models/ used to predict")
    ap.add_argument("--send-to", metavar="USER@IP:FOLDER", help="move the results to this computer afterwards")
    args = ap.parse_args()

    session_dir = ROOT / "images" / args.session
    raw = session_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pi_dir = "/tmp/gopro_" + args.session

    print("== 1/3 capture on the Pi ({} photo(s))".format(args.count), flush=True)
    cmd = "python3 -u {} {} --local {} --interval {}{}{}".format(
        args.pi_script, args.count, pi_dir, args.interval, " --legacy" if args.legacy else "",
        " --camera " + args.camera if args.camera else "")
    capture_ok = subprocess.run(["ssh"] + SSH_OPTS + [args.pi, cmd]).returncode == 0

    print("== 2/3 copy from the Pi to {}".format(raw), flush=True)
    if subprocess.run(["scp", "-q"] + SSH_OPTS + ["{}:{}/*".format(args.pi, pi_dir), str(raw)],
                      stderr=subprocess.DEVNULL).returncode:
        shutil.rmtree(str(session_dir), ignore_errors=True)  # nothing arrived, leave no empty session
        subprocess.run(["ssh"] + SSH_OPTS + [args.pi, "rm -rf " + pi_dir])
        sys.exit("no photos from the Pi (see the capture messages above)")
    subprocess.run(["ssh"] + SSH_OPTS + [args.pi, "rm -rf " + pi_dir])  # copied, so nothing stays on the Pi

    print("== 3/3 predict on this Jetson", flush=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / "process/predict_folder.py"), args.session, "--model", args.model])
    print("results: {}  (comparisons in compare/)".format(session_dir), flush=True)

    if args.send_to and r.returncode == 0:
        print("== send to " + args.send_to, flush=True)
        if subprocess.run(["scp", "-rq"] + SSH_OPTS + [str(session_dir), args.send_to]).returncode == 0:
            shutil.rmtree(str(session_dir))  # the computer has it, nothing stays on the Jetson
            print("sent; removed from the Jetson", flush=True)
        else:
            print("could not send; results stay in {}".format(session_dir), flush=True)
            sys.exit(1)
    if r.returncode or not capture_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
