#!/usr/bin/env python
"""
ForceBody visualization (release 1.3.0, trial format v1.2): render a trial's SKEL body with GRF arrows seated on
the feet.

Two modes:
  png   still image, two views ('view 0 deg' + 'view 60 deg'), one peak single-support frame
  mp4   animation over the contact window ('view 60 deg')

Pose set (each trial ships two, DETAILS.md (Hugging Face dataset repo) "Choosing a pose source"):
  --pose_source pass0   (alias kin, default) poses / trans fitted to the AddB kinematics pass
  --pose_source pass2   (alias dyn)          poses_dyn / trans_dyn fitted to the AddB dynamics pass (repaired target where skel_fit_target_dyn says so)
The trial is read with forcebody_io.load_trial(npz, pose_source), so exactly one pose set is used. If the chosen set is
not usable (pose_usable False, i.e. poses_dyn_usable False for pass2), a warning is printed and the title says
POSE UNUSABLE.

The body uses `skel_gender` and `betas`. The GRF arrow starts at the centre of pressure `cop_aligned_foot` (the
measured CoP in the SKEL frame) and uses `grf_aligned`. The skeleton is coloured by `tau_selfconsistent`, the torque
reproducible from the shipped fields (the AddB native torque `tau_gt_det` is not, see SCHEMA_v1.3.csv). The pelvis is
grey (its entry is the root residual). Foot arrows are green where `grf_foot_ok` and `cop_frame_ok` hold. The Falisse2017
chair channel (`contact_bodies` == 'pelvis') is blue where `grf_frame_ok` holds (CHAIR_RGB, display choice). Arrows on
masked frames are grey. The png frame is chosen among the `grf_frame_ok` frames when the trial has any.

Camera: fixed in the SKEL world frame (y up). 'view 0 deg' looks along -z, 'view 60 deg' is turned 60 deg about the
vertical axis. The views do not follow the body heading. The frame holds the skin and every drawn arrow.

Requirements (VISUALIZATION.md, requirements_visualize.txt): the public SKEL package
(https://github.com/MarilynKeller/SKEL), the SKEL model files (skel_male.pkl, skel_female.pkl, version 1.1.1, from
https://skel.is.tue.mpg.de/) given with --skel_model or the SKEL_MODEL_PATH environment variable, torch, scipy,
pyrender with an EGL-capable driver, trimesh, imageio and Pillow. The rendering helpers are in skel_render.py next to
this file. SKEL forward kinematics runs on GPU if available and on CPU otherwise, only for the frames that are drawn.

Usage ($DATA = the downloaded ForceBody folder):
    export SKEL_MODEL_PATH=/path/to/skel_models
    python visualize.py --npz $DATA/trials/vanderZee2022/S08/vanderZee2022_S08_walk_122.npz --mode png --pose_source pass0
    python visualize.py --npz <path> --mode mp4 --pose_source pass2 --skel_model /path/to/skel_models --out my_clip.mp4

Frames: poses / trans / joints and every *_aligned field are in the SKEL world frame (y up). The raw grf / cop /
contact_torque / markers fields are in the AddB lab frame and are not used for rendering.
"""
import os
import sys
import argparse
import functools

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forcebody_io as fio  # noqa: E402
import skel_render as sr  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHAIR_RGB = (0.20, 0.45, 0.95)   # chair-channel arrow colour (display choice)
PNG_VIEWS = ('view 0 deg', 'view 60 deg')
MP4_VIEW = 'view 60 deg'


@functools.lru_cache(maxsize=None)
def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _fit(txt, width, size=22):
    """Largest font size (22 down to 10) at which txt fits in width."""
    d = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    while size > 10 and d.textlength(txt, font=_font(size)) > width - 12:
        size -= 1
    return size


def _label(img, txt):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = _font(20)
    d.rectangle([0, 0, int(d.textlength(txt, font=font)) + 8, 28], fill=(0, 0, 0))
    d.text((4, 2), txt, fill=(255, 255, 255), font=font)
    return np.array(im)


def _header(img, lines, sizes=None, band=32):
    """Title band above img: one band per line. `sizes` gives the font size per line (default: each line shrunk to
    fit the width). The height grows by 32 px per line, so an even image stays even (H.264)."""
    strips = []
    for k, txt in enumerate(lines):
        size = sizes[k] if sizes else _fit(txt, img.shape[1])
        strip = Image.new('RGB', (img.shape[1], band), (0, 0, 0))
        ImageDraw.Draw(strip).text((6, (band - size) // 2 - 1), txt, fill=(255, 255, 255), font=_font(size))
        strips.append(np.array(strip))
    return np.concatenate(strips + [img], axis=0)


def _chair_channels(d):
    return [c for c, b in enumerate(map(str, d['contact_bodies'])) if b == 'pelvis']


def _load(npz_path, pose_source):
    d = fio.load_trial(npz_path, pose_source)
    if not bool(d['pose_usable']):
        print(f'warning: pose_usable is False (poses_dyn_usable False): the {pose_source} pose set of this trial is not '
              'usable (DETAILS.md). It is drawn anyway and the title says POSE UNUSABLE.', file=sys.stderr)
    return d


def _arrows(d, frames):
    """Per frame in `frames`: list of (cop [3], grf [3], rgb) in the SKEL world frame, one per channel with
    |GRF| >= sr.ARROW_MIN_N. Foot channels (0 right, 1 left) are green where grf_foot_ok and cop_frame_ok hold, the
    chair channel is blue where grf_frame_ok holds, and every other arrow is grey (masked)."""
    grf = np.asarray(d['grf_aligned'])
    cop = np.asarray(d['cop_aligned_foot'])
    frame_ok = np.asarray(d['grf_frame_ok'])
    foot_ok = np.asarray(d['grf_foot_ok']) & np.asarray(d['cop_frame_ok'])
    chair = _chair_channels(d)
    out = []
    for fi in frames:
        arrows = []
        for c in range(grf.shape[1]):
            if np.linalg.norm(grf[fi, c]) < sr.ARROW_MIN_N:
                continue
            ok, rgb = (frame_ok[fi], CHAIR_RGB) if c in chair else (foot_ok[fi, c], sr.GRF_RGB)
            arrows.append((cop[fi, c], grf[fi, c], rgb if ok else sr.MASKED_RGB))
        out.append(arrows)
    return out


def _arrow_points(arrow_lists):
    """CoP points and arrow tips [M,3] (SKEL world frame) of every drawn arrow, for the camera framing."""
    pts = [p for arrows in arrow_lists for cop, grf, _ in arrows for p in (cop, cop + grf * sr.ARROW_SCALE)]
    return np.array(pts).reshape(-1, 3)


def _scene_data(d, frames, model_dir):
    """SKEL meshes and render inputs of `frames`. Every array is indexed by position in `frames` (0..n-1)."""
    g = str(d['skel_gender'])
    sd = sr.skel_meshes(d['betas'], np.asarray(d['poses'])[frames], np.asarray(d['trans'])[frames], g, model_dir,
                        device=DEVICE)
    sd['tau'] = np.asarray(d['tau_selfconsistent'])[frames]
    sd['grf_data'] = _arrows(d, frames)
    sd['lbs'] = sr.load_lbs_weights(model_dir, g)
    return sd


def _panel(sd, i, cam, size, view, tp):
    return sr.render_panel_torque(sd, [i], *cam, size, view, lbs_weights=sd['lbs'], torque_percentiles=tp,
                                  show_grf=True, show_skin=True)


def _title(npz_path, d, fi, gm, arrows):
    """Two title lines: the trial and pose set, then the frame, its largest GRF and the GRF mask state."""
    top = (f'{os.path.basename(npz_path)} [{d["activity_name"]}] {d["pose_source"]} '
           f'({d["pose_fit_target"]})')
    if not bool(d['pose_usable']):
        top += '  POSE UNUSABLE'
    bottom = f'f{fi}  max GRF {gm[fi].max():.0f} N  '
    if not bool(d['grf_frame_ok'][fi]):
        bottom += 'grf_frame_ok False (GRF masked)'
    elif any(rgb == sr.MASKED_RGB for _, _, rgb in arrows):
        bottom += 'grey arrow: grf_foot_ok/cop_frame_ok False'
    else:
        bottom += 'grf_frame_ok'
    return [top, bottom]


def render_png(npz_path, out, pose_source, model_dir):
    d = _load(npz_path, pose_source)
    gm = np.linalg.norm(np.asarray(d['grf']), axis=2)
    cand = np.asarray(d['grf_frame_ok']).astype(bool)
    if not cand.any():
        print('warning: no frame of this trial has grf_frame_ok, so the frame is chosen among all frames and its GRF '
              'is masked', file=sys.stderr)
        cand = np.ones_like(cand)
    ss = np.where(cand & (((gm[:, 0] > 100) & (gm[:, 1] < 10)) | ((gm[:, 1] > 100) & (gm[:, 0] < 10))))[0]
    if len(ss):
        fi = int(ss[np.argmax(gm[ss].max(1))])
    else:
        idx = np.flatnonzero(cand)
        fi = int(idx[np.argmax(gm[idx].max(1))])
    sd = _scene_data(d, [fi], model_dir)
    tp = sr.build_torque_percentiles(sd['tau'], [0])
    cams = sr.compute_camera(sd, [0], 560, 5000, views=PNG_VIEWS, points=_arrow_points(sd['grf_data']))
    panels = [_label(_panel(sd, 0, cams[view], 560, view, tp), view) for view in PNG_VIEWS]
    imageio.imwrite(out, _header(np.concatenate(panels, axis=1), _title(npz_path, d, fi, gm, sd['grf_data'][0])))
    print(f'wrote {out}')


def render_mp4(npz_path, out, pose_source, model_dir, max_frames=150, fps=20, realtime=False):
    d = _load(npz_path, pose_source)
    gm = np.linalg.norm(np.asarray(d['grf']), axis=2)
    contact = np.where(gm.max(1) > 50)[0]
    if len(contact) == 0:
        print('no contact frames')
        return
    frames = list(range(int(contact[0]), int(contact[-1]) + 1))
    step = len(frames) // max_frames + 1 if len(frames) > max_frames else 1
    frames = frames[::step]
    rate = float(d['fps_true_grid'])          # trial frame rate (Hz)
    if realtime:
        fps = rate / step
    sd = _scene_data(d, frames, model_dir)
    local = list(range(len(frames)))
    tp = sr.build_torque_percentiles(sd['tau'], local)
    cam = sr.compute_camera(sd, local, 640, 5000, views=(MP4_VIEW,), points=_arrow_points(sd['grf_data']))[MP4_VIEW]
    titles = [_title(npz_path, d, fi, gm, sd['grf_data'][i]) for i, fi in enumerate(frames)]
    sizes = [min(_fit(t[k], 640) for t in titles) for k in range(2)]   # one font size per line for the whole clip
    w = imageio.get_writer(out, fps=fps, quality=8)
    for i in local:
        img = _panel(sd, i, cam, 640, MP4_VIEW, tp)
        w.append_data(_header(img, titles[i], sizes))
    w.close()
    print(f'wrote {out} ({len(frames)} frames, step {step} over the trial frames at {rate:.1f} Hz, {fps:.2f} fps, '
          f'playback {fps * step / rate:.2f} x real time)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='path to a trial .npz')
    ap.add_argument('--mode', choices=['png', 'mp4'], default='png')
    ap.add_argument('--pose_source', choices=sorted(fio.POSE_SOURCES), default='pass0',
                    help="pose set: 'pass0'/'kin' (poses, default) or 'pass2'/'dyn' (poses_dyn)")
    ap.add_argument('--skel_model', default=None,
                    help='folder with skel_male.pkl and skel_female.pkl (SKEL 1.1.1). Default: $SKEL_MODEL_PATH')
    ap.add_argument('--out', default=None, help='output path (default: <folder>__<trial>_<pose_source>.<mode>)')
    ap.add_argument('--fps', type=int, default=20, help='mp4 frame rate (playback speed varies with the trial)')
    ap.add_argument('--max-frames', type=int, default=150)
    ap.add_argument('--realtime', action='store_true',
                    help='mp4: set the frame rate so that the clip plays in real time (overrides --fps)')
    a = ap.parse_args()
    try:
        model_dir = sr.resolve_skel_model(a.skel_model)
        sr.import_skel()
    except (FileNotFoundError, ImportError) as e:
        sys.exit(f'visualize.py: {e}')
    stem = os.path.splitext(os.path.basename(a.npz))[0]
    parent = os.path.basename(os.path.dirname(a.npz))
    src = fio.canonical_pose_source(a.pose_source)
    out = a.out or f'{parent}__{stem}_{src}.{a.mode}'
    if a.mode == 'png':
        render_png(a.npz, out, src, model_dir)
    else:
        render_mp4(a.npz, out, src, model_dir, a.max_frames, a.fps, a.realtime)


if __name__ == '__main__':
    main()
