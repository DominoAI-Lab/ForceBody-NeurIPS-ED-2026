#!/usr/bin/env python
"""
ForceBody visualization (release 1.3.0, trial format v1.2): render a trial's SKEL body with GRF arrows seated on
the feet.

Two modes:
  png   still image, two views (front + side), one peak single-support frame
  mp4   animation over the contact window (side view)

Pose set (each trial ships two, DETAILS.md "Choosing a pose source"):
  --pose_source pass0   (alias kin, default) poses / trans fitted to the AddB kinematics pass
  --pose_source pass2   (alias dyn)          poses_dyn / trans_dyn fitted to the AddB dynamics pass (repaired target where skel_fit_target_dyn says so)
The trial is read with forcebody_io.load_trial(npz, pose_source), so exactly one pose set is used.

The body uses `skel_gender` and `betas`. The GRF arrow starts at the centre of pressure `cop_aligned_foot` (the
measured CoP in the SKEL frame) and uses `grf_aligned`. The skeleton is coloured by `tau_selfconsistent`, the torque
reproducible from the shipped fields (the AddB native torque `tau_gt_det` is not, see SCHEMA_v1.3.csv). Foot arrows
are green. The Falisse2017 chair channel (`contact_bodies` == 'pelvis') is drawn in blue (CHAIR_RGB, display choice).

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

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forcebody_io as fio  # noqa: E402
import skel_render as sr  # noqa: E402
import torch  # noqa: E402
import imageio  # noqa: E402
import pyrender  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHAIR_RGB = (0.20, 0.45, 0.95)   # chair-channel arrow colour (display choice)


def _font(size):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _label(img, txt):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = _font(20)
    d.rectangle([0, 0, int(d.textlength(txt, font=font)) + 8, 28], fill=(0, 0, 0))
    d.text((4, 2), txt, fill=(255, 255, 255), font=font)
    return np.array(im)


def _header(img, txt, band=32):
    strip = Image.new('RGB', (img.shape[1], band), (0, 0, 0))
    d = ImageDraw.Draw(strip)
    size = 22
    while size > 10 and d.textlength(txt, font=_font(size)) > img.shape[1] - 12:
        size -= 1
    d.text((6, 4), txt, fill=(255, 255, 255), font=_font(size))
    return np.concatenate([np.array(strip), img], axis=0)


def _chair_channels(d):
    return [c for c, b in enumerate(map(str, d['contact_bodies'])) if b == 'pelvis']


def _grf_data(cop, grf, skip=()):
    contact = np.linalg.norm(grf, axis=2) > 0
    contact[:, list(skip)] = False          # chair channel drawn separately (_panel)
    return [(contact[i], grf[i], cop[i]) for i in range(grf.shape[0])]


def _panel(sd, i, cam, size, view, tp, chair):
    """render_panel_torque of local frame i plus the chair-channel arrows in CHAIR_RGB (added to the same scene)."""
    extra = []
    for c in chair:
        F = np.array(sd['grf_raw'][i][c], float)
        if np.linalg.norm(F) > 0:
            F[1] = -F[1]                        # same display flip as render_panel_torque
            arrow = sr.make_grf_arrow(sr.prep_vertices(sd['cop_raw'][i][c:c + 1].copy(), cam[2])[0], F)
            if arrow is not None:
                extra.append((arrow, pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE', baseColorFactor=(*CHAIR_RGB, 1.0))))
    return sr.render_panel_torque(sd, [i], *cam, size, view, lbs_weights=sd['lbs'], torque_percentiles=tp,
                                  show_grf=True, show_skin=True, extra_meshes=extra)


def _scene_data(d, frames, model_dir):
    """SKEL meshes and render inputs of `frames`. Every array is indexed by position in `frames` (0..n-1)."""
    g = str(d['skel_gender'])
    sd = sr.skel_meshes(d['betas'], np.asarray(d['poses'])[frames], np.asarray(d['trans'])[frames], g, model_dir,
                        device=DEVICE)
    sd['tau'] = np.asarray(d['tau_selfconsistent'])[frames]
    grf_al = np.asarray(d['grf_aligned'])[frames]
    cop = np.asarray(d['cop_aligned_foot'])[frames]
    sd['grf_data'] = _grf_data(cop, grf_al, _chair_channels(d))
    sd['grf_raw'], sd['cop_raw'] = grf_al, cop
    sd['lbs'] = sr.load_lbs_weights(model_dir, g)
    return sd


def _title(npz_path, d, fi, gm):
    return (f'{os.path.basename(npz_path)} [{d["activity_name"]}] {d["pose_source"]} '
            f'({d["pose_fit_target"]}) f{fi} {gm[fi].max():.0f}N')


def render_png(npz_path, out, pose_source, model_dir):
    d = fio.load_trial(npz_path, pose_source)
    gm = np.linalg.norm(np.asarray(d['grf']), axis=2)
    ss = np.where(((gm[:, 0] > 100) & (gm[:, 1] < 10)) | ((gm[:, 1] > 100) & (gm[:, 0] < 10)))[0]
    fi = int(ss[np.argmax(gm[ss].max(1))]) if len(ss) else int(np.argmax(gm.max(1)))
    sd = _scene_data(d, [fi], model_dir)
    tp = sr.build_torque_percentiles(sd['tau'], [0])
    panels = []
    for view in ['front', 'side60d']:
        cam = sr.compute_camera(sd, [0], 560, 5000)
        img = _panel(sd, 0, cam, 560, view, tp, _chair_channels(d))
        panels.append(_label(img, view))
    imageio.imwrite(out, _header(np.concatenate(panels, axis=1), _title(npz_path, d, fi, gm)))
    print(f'wrote {out}')


def render_mp4(npz_path, out, pose_source, model_dir, max_frames=150, fps=20):
    d = fio.load_trial(npz_path, pose_source)
    gm = np.linalg.norm(np.asarray(d['grf']), axis=2)
    contact = np.where(gm.max(1) > 50)[0]
    if len(contact) == 0:
        print('no contact frames')
        return
    frames = list(range(int(contact[0]), int(contact[-1]) + 1))
    if len(frames) > max_frames:
        frames = frames[::len(frames) // max_frames + 1]
    sd = _scene_data(d, frames, model_dir)
    local = list(range(len(frames)))
    tp = sr.build_torque_percentiles(sd['tau'], local)
    cam = sr.compute_camera(sd, local, 640, 5000)
    w = imageio.get_writer(out, fps=fps, quality=8)
    for i, fi in enumerate(frames):
        img = _panel(sd, i, cam, 640, 'side60d', tp, _chair_channels(d))
        w.append_data(_label(img, _title(npz_path, d, fi, gm)))
    w.close()
    print(f'wrote {out} ({len(frames)} frames)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='path to a trial .npz')
    ap.add_argument('--mode', choices=['png', 'mp4'], default='png')
    ap.add_argument('--pose_source', choices=sorted(fio.POSE_SOURCES), default='pass0',
                    help="pose set: 'pass0'/'kin' (poses, default) or 'pass2'/'dyn' (poses_dyn)")
    ap.add_argument('--skel_model', default=None,
                    help='folder with skel_male.pkl and skel_female.pkl (SKEL 1.1.1). Default: $SKEL_MODEL_PATH')
    ap.add_argument('--out', default=None, help='output path (default: <folder>__<trial>_<pose_source>.<mode>)')
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--max-frames', type=int, default=150)
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
        render_mp4(a.npz, out, src, model_dir, a.max_frames, a.fps)


if __name__ == '__main__':
    main()
