#!/usr/bin/env python
"""
ForceBody v1.2 visualization: render a trial's SKEL body with GRF arrows seated on the feet.

Two modes:
  png   still image, two views (front + side), one peak single-support frame
  mp4   animation over the contact window (side view)

Pose set (v1.2 ships two, README "Choosing a pose source"):
  --pose_source pass0   (alias kin, default) poses / trans fitted to the AddB kinematics pass
  --pose_source pass2   (alias dyn)          poses_dyn / trans_dyn fitted to the AddB dynamics pass (repaired target where skel_fit_target_dyn says so)
The trial is read with forcebody_io.load_trial(npz, pose_source), so exactly one pose set is used.

The body uses `skel_gender` and `betas`. The GRF arrow starts at the centre of pressure `cop_aligned_foot` (the
measured CoP in the SKEL frame) and uses `grf_aligned`; the skeleton is coloured by `tau_selfconsistent` (the torque
reproducible from the shipped fields; `tau_gt_det` is not used, see README known issues). Foot arrows use the
renderer's default green; the Falisse2017 chair channel (`contact_bodies` == 'pelvis') is drawn in its own colour
(blue, CHAIR_RGB; display choice, OURS).

Requirements: the addb2skel package importable (install it, or set ADDB2SKEL_ROOT to the folder that contains it) and its
render utilities (`addb2skel.utils.render`), an EGL-capable renderer, and the SKEL body model assets addb2skel uses.
addb2skel is not distributed with ForceBody (README, Known issues); the data files themselves need only numpy.
SKEL forward kinematics runs on GPU if available and on CPU otherwise.

Usage:
    python visualize.py --npz trials/vanderZee2022/S08/vanderZee2022_S08_walk_122.npz --mode png --pose_source pass0
    python visualize.py --npz <path> --mode mp4 --pose_source pass2 --out my_clip.mp4

Frames: poses / trans / joints and every *_aligned field are in the SKEL world frame (y up). The raw grf / cop /
contact_torque / markers fields are in the AddB lab frame and are not used for rendering.
"""
import os
import sys
import argparse
import tempfile

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forcebody_io as fio  # noqa: E402

if os.environ.get('ADDB2SKEL_ROOT'):
    sys.path.insert(0, os.environ['ADDB2SKEL_ROOT'])
try:
    from addb2skel.utils.render import (load_skel_data, compute_camera, render_panel_torque,  # noqa: E402
                                        build_torque_percentiles, load_lbs_weights)
    import addb2skel.utils.render as _R  # noqa: E402
except ImportError as e:  # pragma: no cover
    sys.exit(f'addb2skel is not importable ({e}); install it or set ADDB2SKEL_ROOT to the folder containing it')
import torch  # noqa: E402
import imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHAIR_RGB = (0.20, 0.45, 0.95)   # chair-channel arrow colour (display choice, OURS)


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _label(img, txt):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(txt) * 11 + 8, 28], fill=(0, 0, 0))
    d.text((4, 2), txt, fill=(255, 255, 255), font=_font(20))
    return np.array(im)


def _header(img, txt, band=32):
    strip = Image.new('RGB', (img.shape[1], band), (0, 0, 0))
    ImageDraw.Draw(strip).text((6, 4), txt, fill=(255, 255, 255), font=_font(22))
    return np.concatenate([np.array(strip), img], axis=0)


def _chair_channels(d):
    return [c for c, b in enumerate(map(str, d['contact_bodies'])) if b == 'pelvis']


def _grf_data(cop, grf, skip=()):
    contact = np.linalg.norm(grf, axis=2) > 0
    contact[:, list(skip)] = False          # chair channel drawn separately (_panel)
    return [(contact[i], grf[i], cop[i]) for i in range(grf.shape[0])]


def _panel(sd, fi, cam, size, view, lbs, tp, chair):
    """render_panel_torque plus the chair-channel arrows in CHAIR_RGB (added to the same scene)."""
    import pyrender
    extra = []
    for c in chair:
        F = np.array(sd['grf_raw'][fi][c], float)
        if np.linalg.norm(F) > 0:
            F[1] = -F[1]                        # same display flip as render_panel_torque
            arrow = _R.make_grf_arrow(_R.prep_vertices(sd['cop_raw'][fi][c:c + 1].copy(), cam[2])[0], F)
            if arrow is not None:
                extra.append((arrow, pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE', baseColorFactor=(*CHAIR_RGB, 1.0))))
    orig = _R.render_scene
    _R.render_scene = lambda meshes, *a, **k: orig(meshes + extra, *a, **k)
    try:
        return render_panel_torque(sd, [fi], *cam, size, view, lbs_weights=lbs, torque_percentiles=tp, show_grf=True,
                                   show_skin=True)
    finally:
        _R.render_scene = orig


def _prepare(npz_path, pose_source):
    d = fio.load_trial(npz_path, pose_source)
    g = str(d['skel_gender'])
    # load_skel_data reads betas / poses / trans from a file: hand it the selected pose set only
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, 'pose.npz')
        np.savez(tmp, betas=d['betas'], poses=d['poses'], trans=d['trans'])
        sd = load_skel_data(tmp, g, device=DEVICE, batch_size=64)
    sd['gender'] = g
    sd['tau'] = np.asarray(d['tau_selfconsistent'])
    grf_al = np.asarray(d['grf_aligned'])
    cop = np.asarray(d['cop_aligned_foot'])
    gm = np.linalg.norm(np.asarray(d['grf']), axis=2)
    _, lbs = load_lbs_weights(None, g)
    return d, sd, grf_al, cop, gm, lbs


def _title(npz_path, d, fi, gm):
    return (f'{os.path.basename(npz_path)} [{d["activity_name"]}] {d["pose_source"]} '
            f'({d["pose_fit_target"]}) f{fi} {gm[fi].max():.0f}N')


def render_png(npz_path, out, pose_source):
    d, sd, grf_al, cop, gm, lbs = _prepare(npz_path, pose_source)
    ss = np.where(((gm[:, 0] > 100) & (gm[:, 1] < 10)) | ((gm[:, 1] > 100) & (gm[:, 0] < 10)))[0]
    fi = int(ss[np.argmax(gm[ss].max(1))]) if len(ss) else int(np.argmax(gm.max(1)))
    tp = build_torque_percentiles(sd['tau'], [fi])
    sd['grf_data'] = _grf_data(cop, grf_al, _chair_channels(d))
    sd['grf_raw'], sd['cop_raw'] = grf_al, cop
    panels = []
    for view in ['front', 'side60d']:
        cam = compute_camera(sd, [fi], 560, 5000)
        img = _panel(sd, fi, cam, 560, view, lbs, tp, _chair_channels(d))
        panels.append(_label(img, view))
    imageio.imwrite(out, _header(np.concatenate(panels, axis=1), _title(npz_path, d, fi, gm)))
    print(f'wrote {out}')


def render_mp4(npz_path, out, pose_source, max_frames=150, fps=20):
    d, sd, grf_al, cop, gm, lbs = _prepare(npz_path, pose_source)
    contact = np.where(gm.max(1) > 50)[0]
    if len(contact) == 0:
        print('no contact frames')
        return
    frames = list(range(int(contact[0]), int(contact[-1]) + 1))
    if len(frames) > max_frames:
        frames = frames[::len(frames) // max_frames + 1]
    tp = build_torque_percentiles(sd['tau'], frames)
    cam = compute_camera(sd, frames, 640, 5000)
    sd['grf_data'] = _grf_data(cop, grf_al, _chair_channels(d))
    sd['grf_raw'], sd['cop_raw'] = grf_al, cop
    w = imageio.get_writer(out, fps=fps, quality=8)
    for fi in frames:
        img = _panel(sd, fi, cam, 640, 'side60d', lbs, tp, _chair_channels(d))
        w.append_data(_label(img, _title(npz_path, d, fi, gm)))
    w.close()
    print(f'wrote {out} ({len(frames)} frames)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='path to a trial .npz')
    ap.add_argument('--mode', choices=['png', 'mp4'], default='png')
    ap.add_argument('--pose_source', choices=sorted(fio.POSE_SOURCES), default='pass0',
                    help="pose set: 'pass0'/'kin' (poses, default) or 'pass2'/'dyn' (poses_dyn)")
    ap.add_argument('--out', default=None, help='output path (default: <folder>__<trial>_<pose_source>.<mode>)')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--max-frames', type=int, default=150)
    a = ap.parse_args()
    stem = os.path.splitext(os.path.basename(a.npz))[0]
    parent = os.path.basename(os.path.dirname(a.npz))
    src = fio.canonical_pose_source(a.pose_source)
    out = a.out or f'{parent}__{stem}_{src}.{a.mode}'
    if a.mode == 'png':
        render_png(a.npz, out, src)
    else:
        render_mp4(a.npz, out, src, a.max_frames, a.fps)


if __name__ == '__main__':
    main()
