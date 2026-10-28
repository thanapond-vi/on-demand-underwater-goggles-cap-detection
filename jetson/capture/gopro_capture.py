"""
gopro_capture.py - runs on the Raspberry Pi Zero (Python 3 standard library + ssh/scp)

Tells the GoPro (over its Wi-Fi, HTTP API at 10.5.5.9) to take N photos and sends each one
straight to the Jetson's images/<session>/raw/ over the LAN. Nothing is predicted here and
nothing is kept on the Pi: a photo is deleted from the Pi once the Jetson has it.

Usage:
    python3 gopro_capture.py 10                       # 10 photos, session = date-time (e.g. 20261018_1305)
    python3 gopro_capture.py 10 --session test1       # store in images/test1/raw on the Jetson
    python3 gopro_capture.py 20 --interval 5          # one photo every 5 s
    python3 gopro_capture.py 10 --legacy              # HERO5-HERO8 (older API); default is Open GoPro (HERO9+)
    python3 gopro_capture.py 10 --local /tmp/gopro_x  # only save on the Pi; the Jetson pulls them
                                                      # (used by capture_predict.py run on the Jetson)
If the Pi is not on the GoPro Wi-Fi yet, the Open GoPro SDK (open_gopro, already on the Pi) connects once over
Bluetooth, turns on the camera Wi-Fi and joins it (needs sudo: password from $GOPRO_SUDO_PASSWORD or
~/.gopro_sudo_password), then all photos are taken in that one connection. --camera 8488 picks one camera.
Sending mode also needs an SSH key from the Pi to the Jetson (ssh-copy-id user@192.168.2.1).
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

GOPRO = "http://10.5.5.9"
JETSON = "user@192.168.2.1"            # Jetson eth0 (LAN to the Pi); change with --jetson
JETSON_IMAGES = "pool_detect/images"    # photos go to <this>/<session>/raw/; change with --dest
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
PHOTO_EXT = (".jpg", ".jpeg")

API = {  # Open GoPro (HERO9+) vs legacy (HERO5-HERO8)
    "open": {"list": ":8080/gopro/media/list", "shutter": ":8080/gopro/camera/shutter/start",
             "photo_mode": ":8080/gopro/camera/presets/set_group?id=1001"},
    "legacy": {"list": ":8080/gp/gpMediaList", "shutter": "/gp/gpControl/command/shutter?p=1",
               "photo_mode": "/gp/gpControl/command/mode?p=1"},
}


def get(path, timeout=10):
    with urllib.request.urlopen(GOPRO + path, timeout=timeout) as r:
        return r.read()


def list_photos(api):
    """Return {'100GOPRO/GOPR0001.JPG', ...} for every photo on the SD card."""
    media = json.loads(get(api["list"]).decode())
    return {d["d"] + "/" + f["n"] for d in media.get("media", []) for f in d.get("fs", [])
            if f["n"].lower().endswith(PHOTO_EXT)}


def take_photo(api, timeout=15):
    """Press the shutter and return the name of the new photo once it is saved on the SD card."""
    before = list_photos(api)
    get(api["shutter"])
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        try:
            new = list_photos(api) - before
        except OSError:  # camera busy while writing the photo
            continue
        if new:
            return sorted(new)[-1]
    raise TimeoutError("no new photo {} s after the shutter".format(timeout))


def download(name, folder):
    dst = Path(folder) / name.replace("/", "_")
    with urllib.request.urlopen(GOPRO + ":8080/videos/DCIM/" + name, timeout=60) as r, open(str(dst), "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    return dst


def send(path, session, jetson, dest):
    """Copy to the Jetson; on success delete the Pi's copy. Returns True if sent."""
    r = subprocess.run(["scp", "-q"] + SSH_OPTS + [str(path), "{}:{}/{}/raw/".format(jetson, dest, session)])
    if r.returncode == 0:
        path.unlink()
    return r.returncode == 0


def camera_reachable(api):
    try:
        get(api["list"], timeout=3)
        return True
    except OSError:
        return False


def sudo_password():
    pw = os.environ.get("GOPRO_SUDO_PASSWORD")
    f = Path.home() / ".gopro_sudo_password"
    if not pw and f.exists():
        pw = f.read_text().strip()
    return pw


async def capture_sdk(args, out):
    """Connect with the Open GoPro SDK (Bluetooth -> camera Wi-Fi) once and take all photos in that session."""
    from open_gopro import WirelessGoPro, constants, proto

    target = re.compile(".*" + re.escape(args.camera)) if args.camera else None
    print("connecting to the GoPro over Bluetooth + Wi-Fi (can take a minute)...", flush=True)
    taken = 0
    async with WirelessGoPro(target, wifi_interface=args.wifi_interface, sudo_password=sudo_password()) as gopro:
        await gopro.http_command.load_preset_group(group=proto.EnumPresetGroup.PRESET_GROUP_ID_PHOTO)
        for i in range(args.count):
            t0 = time.time()
            try:
                before = {f.filename for f in (await gopro.http_command.get_media_list()).data.files}
                await gopro.http_command.set_shutter(shutter=constants.Toggle.ENABLE)
                new = set()
                while not new and time.time() - t0 < 15:
                    await asyncio.sleep(0.5)
                    new = {f.filename for f in (await gopro.http_command.get_media_list()).data.files} - before
                    new = {n for n in new if n.lower().endswith(PHOTO_EXT)}
                if not new:
                    raise TimeoutError("no new photo 15 s after the shutter")
                name = sorted(new)[-1]
                dst = out / name.replace("/", "_")
                await gopro.http_command.download_file(camera_file=name, local_file=dst)
                taken += 1
                print("[{}/{}] {}".format(i + 1, args.count, dst.name), flush=True)
            except Exception as e:  # keep going with the next photo
                print("[{}/{}] failed: {!r}".format(i + 1, args.count, e), flush=True)
            if i + 1 < args.count:
                await asyncio.sleep(max(0.0, args.interval - (time.time() - t0)))
    return taken


def capture_local(api, args):
    """Take the photos into args.local only; whoever started us copies them away and deletes the folder."""
    out = Path(args.local)
    out.mkdir(parents=True, exist_ok=True)
    if not camera_reachable(api):
        try:
            import open_gopro  # noqa: F401
        except ImportError:
            sys.exit("GoPro not reachable at 10.5.5.9: join the Pi to the GoPro Wi-Fi (or install open_gopro)")
        try:
            taken = asyncio.run(capture_sdk(args, out))
        except Exception as e:
            sys.exit("could not connect to the GoPro: {!r} (camera on? Bluetooth/Wi-Fi in range? sudo password set?)".format(e))
        print("done: {}/{} photo(s) in {}".format(taken, args.count, out), flush=True)
        if taken < args.count:
            sys.exit(1)
        return
    get(api["photo_mode"])
    time.sleep(1)  # mode switch takes a moment
    taken = 0
    for i in range(args.count):
        t0 = time.time()
        try:
            path = download(take_photo(api), out)
            taken += 1
            print("[{}/{}] {}".format(i + 1, args.count, path.name), flush=True)
        except OSError as e:  # includes TimeoutError: GoPro asleep / Wi-Fi dropped
            print("[{}/{}] failed: {}".format(i + 1, args.count, e), flush=True)
        if i + 1 < args.count:
            time.sleep(max(0.0, args.interval - (time.time() - t0)))
    print("done: {}/{} photo(s) in {}".format(taken, args.count, out), flush=True)
    if taken < args.count:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("count", type=int, help="number of photos to take")
    ap.add_argument("--session", default=time.strftime("%Y%m%d_%H%M"), help="folder name on the Jetson")
    ap.add_argument("--interval", type=float, default=0, help="seconds between shots (0 = as fast as possible)")
    ap.add_argument("--legacy", action="store_true", help="HERO5-HERO8 API")
    ap.add_argument("--jetson", default=JETSON, help="user@ip of the Jetson as seen from the Pi")
    ap.add_argument("--dest", default=JETSON_IMAGES, help="images folder on the Jetson (relative to its home)")
    ap.add_argument("--local", help="save photos in this Pi folder instead of sending them (the Jetson pulls them)")
    ap.add_argument("--camera", help="last digits of the GoPro name, e.g. 8488 (SDK connection only)")
    ap.add_argument("--wifi-interface", default="wlan0", help="Pi Wi-Fi used for the GoPro (SDK connection only)")
    args = ap.parse_args()
    api = API["legacy" if args.legacy else "open"]
    if args.local:
        return capture_local(api, args)

    if not camera_reachable(api):
        sys.exit("GoPro not reachable at 10.5.5.9: join the Pi to the GoPro Wi-Fi, or use --local (SDK connects itself)")
    subprocess.run(["ssh"] + SSH_OPTS + [args.jetson, "mkdir -p {}/{}/raw".format(args.dest, args.session)], check=True)
    get(api["photo_mode"])
    time.sleep(1)  # mode switch takes a moment
    tmp = Path(tempfile.mkdtemp(prefix="gopro_"))  # a photo only sits here until the Jetson has it
    print("taking {} photo(s) -> {}:{}/{}/raw".format(args.count, args.jetson, args.dest, args.session), flush=True)

    unsent, taken = [], 0
    for i in range(args.count):
        t0 = time.time()
        try:
            path = download(take_photo(api), tmp)
        except OSError as e:  # includes TimeoutError: GoPro asleep / Wi-Fi dropped
            print("[{}/{}] failed: {}".format(i + 1, args.count, e), flush=True)
            continue
        taken += 1
        ok = send(path, args.session, args.jetson, args.dest)
        if not ok:
            unsent.append(path)
        print("[{}/{}] {} {}".format(i + 1, args.count, path.name, "sent" if ok else "NOT sent yet"), flush=True)
        if i + 1 < args.count:
            time.sleep(max(0.0, args.interval - (time.time() - t0)))

    unsent = [p for p in unsent if not send(p, args.session, args.jetson, args.dest)]  # one retry at the end
    if unsent:
        sys.exit("{} photo(s) could not reach the Jetson; kept in {} (also still on the GoPro SD card)"
                 .format(len(unsent), tmp))
    tmp.rmdir()
    print("done: {}/{} photo(s) stored on the Jetson in {}/{}/raw".format(taken, args.count, args.dest, args.session), flush=True)


if __name__ == "__main__":
    main()
