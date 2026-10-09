"""
Series runner.

    python -m backrooms.render ep01_level254 --still 40,90            # stills (1080p)
    python -m backrooms.render ep01_level254 --res 4k --still 90      # 4K stills
    python -m backrooms.render ep01_level254 --all                    # video + audio + mux, 1080p
    python -m backrooms.render ep01_level254 --all --res 4k           # same at 3840x2160
    python -m backrooms.render ep01_level254 --audio --mux            # re-mix sound only (e.g. after voice arrives)
    python -m backrooms.render ep01_level254 --web                    # small 1080p delivery copy (<100 MB)

Outputs go to out/<episode>/.
"""
import argparse
import importlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--res", default="1080", choices=["720", "1080", "4k"])
    ap.add_argument("--still", default="")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--web", action="store_true", help="two-pass ~3 Mbps 1080p delivery copy")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--crf", type=int, default=None)
    args = ap.parse_args()

    os.environ["BR_RES"] = args.res
    sys.path.insert(0, ROOT)
    ep = args.episode.strip("/").split("/")[-1]
    film = importlib.import_module(f"episodes.{ep}.film")
    ep_dir = os.path.join(ROOT, "episodes", ep)
    out_dir = os.path.join(ROOT, "out", ep)
    os.makedirs(out_dir, exist_ok=True)
    tag = {"720": "720p", "1080": "1080p", "4k": "2160p"}[args.res]
    video = os.path.join(out_dir, f"{ep}_{tag}_picture.mp4")
    audio = os.path.join(out_dir, f"{ep}_audio.wav")
    final = os.path.join(out_dir, f"{ep}_{tag}.mp4")

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    if args.still:
        from PIL import Image
        from backrooms.core import FPS
        sd = os.path.join(out_dir, f"stills_{tag}")
        os.makedirs(sd, exist_ok=True)
        for s in args.still.split(","):
            t = float(s)
            t0 = time.time()
            fr = film.render_frame(int(round(t * FPS)))
            p = os.path.join(sd, f"t{t:06.1f}.jpg")
            Image.fromarray(fr).save(p, quality=90)
            print(p, f"{time.time() - t0:.2f}s", flush=True)

    if args.video or args.all:
        render_video(film, ff, video, args)

    if args.audio or args.all:
        build_audio(ep, ep_dir, audio, film.DURATION)

    if args.mux or args.all:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", video, "-i", audio, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", final], check=True)
        print("final:", final)

    if args.web:
        src = os.path.join(out_dir, f"{ep}_1080p.mp4")
        dst = os.path.join(out_dir, f"{ep}_1080p_web.mp4")
        passlog = os.path.join(out_dir, "x264pass")
        common = ["-c:v", "libx264", "-preset", "slow", "-b:v", "3000k", "-passlogfile", passlog]
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, *common, "-pass", "1", "-an", "-f", "null", os.devnull], check=True)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, *common, "-maxrate", "5500k", "-bufsize", "11000k",
                        "-pass", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", dst], check=True)
        for f in os.listdir(out_dir):
            if f.startswith("x264pass"):
                os.remove(os.path.join(out_dir, f))
        print("web:", dst, f"{os.path.getsize(dst) / 1e6:.1f} MB")


_FILM = None


def _init(epname, res):
    global _FILM
    os.environ["BR_RES"] = res
    sys.path.insert(0, ROOT)
    _FILM = importlib.import_module(f"episodes.{epname}.film")


def _worker(fi):
    return fi, _FILM.render_frame(fi).tobytes()


def render_video(film, ff, path, args):
    from multiprocessing import Pool
    from backrooms.core import W, H, FPS
    f0 = int(args.start * FPS)
    f1 = int((args.end if args.end is not None else film.DURATION) * FPS)
    crf = args.crf if args.crf is not None else (17 if H > 1500 else 19)
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-tune", "grain", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    ep = film.__name__.split(".")[1]
    with Pool(args.jobs, initializer=_init, initargs=(ep, args.res)) as pool:
        for n, (fi, buf) in enumerate(pool.imap(_worker, range(f0, f1), chunksize=2)):
            p.stdin.write(buf)
            if n % 48 == 0:
                el = time.time() - t0
                print(f"frame {fi}/{f1}  {el:.0f}s elapsed  eta {el / (n + 1) * (f1 - f0 - n - 1):.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    print("picture:", path, f"{time.time() - t0:.0f}s", flush=True)


def build_audio(ep, ep_dir, path, duration):
    import numpy as np
    from backrooms import sound
    from backrooms.voice import mix_voice
    score = importlib.import_module(f"episodes.{ep}.audio")
    music = sound.master(score.build(), duration, fade=0.01)
    mixed, nv = mix_voice(music, ep_dir)
    # soft ceiling above 0.85
    a = np.abs(mixed)
    lim = np.where(a > 0.85, 0.85 + 0.13 * np.tanh((a - 0.85) / 0.13), a)
    mixed = np.sign(mixed) * lim
    mixed *= sound.env_curve([(0, 1), (duration - 1.5, 1), (duration, 0)], mixed.shape[1])
    sound.write_wav(path, mixed)
    print("audio:", path, f"(voice lines: {nv})")


if __name__ == "__main__":
    main()
