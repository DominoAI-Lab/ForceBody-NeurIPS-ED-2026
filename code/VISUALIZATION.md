# Visualization

`visualize.py` renders the SKEL body of one trial with GRF arrows at the centre of pressure. It needs `forcebody_io.py` and `skel_render.py` from this folder, the public SKEL package and the SKEL model files.

## Install

1. Install the Python packages from this folder. Use Python 3.9 to 3.12.

   ```bash
   pip install -r requirements_visualize.txt
   ```

   The file lists numpy (below 2), scipy, torch, pyrender, trimesh, PyOpenGL, imageio with ffmpeg, Pillow and the SKEL package from https://github.com/MarilynKeller/SKEL (pinned to commit `c32cf16`, the SKEL version used to build ForceBody). The pinned SKEL commit requires moderngl-window 2.4.6, which needs numpy below 2. numpy 1.26 has no wheels for Python 3.13, so use Python 3.12 or older.

   `visualize.py` uses only the SKEL model loader, which imports torch, numpy and scipy. If the other SKEL dependencies do not install, install every other package first and then SKEL alone without its dependencies.

   ```bash
   grep -v '^skel' requirements_visualize.txt > req_noskel.txt
   pip install -r req_noskel.txt
   pip install --no-deps "git+https://github.com/MarilynKeller/SKEL.git@c32cf16581295bff19399379efe5b776d707cd95"
   ```

2. Download the SKEL model files. Register at https://skel.is.tue.mpg.de/ and download the SKEL model (version 1.1.1, `skel_male.pkl` and `skel_female.pkl`) under the SKEL license. ForceBody does not redistribute them. The download is `skel_models_v1.1.zip`. It unpacks to a folder `skel_models_v1.1/` that holds the two files. Use that folder in step 3.

3. Give the model folder to `visualize.py` with `--skel_model /path/to/folder` or with the environment variable `SKEL_MODEL_PATH`. The flag takes precedence.

Rendering is offscreen with pyrender. `visualize.py` sets `PYOPENGL_PLATFORM=egl` unless the variable is already set, so it needs an EGL-capable driver (a display is not needed). The SKEL forward pass runs on the GPU when torch finds one and on the CPU otherwise, and only for the frames that are drawn.

Tested on Linux with EGL in two environments, both with SKEL commit `c32cf16`.

- Python 3.9, torch 2.8.0, pyrender 0.1.45, PyOpenGL 3.1.0, trimesh 3.23.5 and 4.12.2, imageio 2.37.2 with imageio-ffmpeg 0.6.0, Pillow 11.3.0, numpy 1.26.4 and scipy 1.13.1.
- A new Python 3.12 venv (its own pip 25.0.1) after `pip install -r requirements_visualize.txt`: torch 2.15.0, pyrender 0.1.45, PyOpenGL 3.1.0, trimesh 5.2.0, imageio 2.38.1 with imageio-ffmpeg 0.7.0, Pillow 13.0.0, numpy 1.26.4 and scipy 1.17.1.

## Usage

`DATA` is the dataset folder from Quick start step 1 in the main README. Run the commands in this folder.

```bash
DATA=/path/to/ForceBody
export SKEL_MODEL_PATH=/path/to/skel_models
python visualize.py --npz $DATA/trials/Moore2015/S10/Moore2015_S10_walk_014.npz --mode png --pose_source pass0
python visualize.py --npz $DATA/trials/Moore2015/S10/Moore2015_S10_walk_014.npz --mode mp4 --pose_source pass2
```

| option | default | meaning |
|---|---|---|
| `--npz` | required | trial file |
| `--mode` | `png` | `png` (still image) or `mp4` (animation) |
| `--pose_source` | `pass0` | `pass0` (alias `kin`) or `pass2` (alias `dyn`) |
| `--skel_model` | `$SKEL_MODEL_PATH` | folder with `skel_male.pkl` and `skel_female.pkl` |
| `--out` | `<subject>__<trial>_<pose_source>.<mode>` in the current folder | output file |
| `--fps` | 20 | mp4 frame rate. The playback speed then depends on the trial (see Outputs). |
| `--max-frames` | 150 | mp4 frame limit. A longer contact window is subsampled with a constant step. |
| `--realtime` | off | mp4: frame rate = trial frame rate / subsampling step, so the clip plays in real time. Overrides `--fps`. |

## Outputs

- `png`: one 1120 x 624 image with two 560 x 560 panels, `view 0 deg` and `view 60 deg`, under a two-line title band. The frame is chosen among the frames with `grf_frame_ok`. Only a trial without such a frame uses all frames, and `visualize.py` prints a warning. The frame is the single-support frame with the largest GRF (one foot above 100 N, the other below 10 N). Without such a frame it is the frame with the largest GRF on any channel.
- `mp4`: a 640 x 704 H.264 clip (one 640 x 640 `view 60 deg` panel under a two-line title band) over the contact window, from the first to the last frame with a GRF above 50 N on any channel, with at most `--max-frames` frames.
- mp4 playback speed: `visualize.py` prints the subsampling step and the playback speed (frame rate x step / trial frame rate). The speed changes from trial to trial. At the default 20 fps, Moore2015_S10_walk_014 (20 s window, step 14) plays at 2.8 x real time and Uhlrich2023_S07_dropjump_001 (1.3 s, step 1) at 0.2 x. Use `--realtime` for 1.0 x.
- Title line 1 gives the trial file, the activity, the pose source and the fit target (`skel_fit_target` or `skel_fit_target_dyn`). It ends with `POSE UNUSABLE` if the chosen pose set is not usable (`pose_usable` False, which means `poses_dyn_usable` False for `pass2`). `visualize.py` then also prints a warning.
- Title line 2 gives the frame index, the largest GRF of that frame in N and the GRF mask state of the frame. The state is `grf_frame_ok`, `grey arrow: grf_foot_ok/cop_frame_ok False` (the frame is valid but one foot is masked) or `grf_frame_ok False (GRF masked)`.

## What is drawn

- `--pose_source pass0` (default, alias `kin`) draws `poses` / `trans`. `pass2` (alias `dyn`) draws `poses_dyn` / `trans_dyn`. The trial is read with `forcebody_io.load_trial`, so only one pose set is used.
- The body uses `skel_gender` and `betas`. The skin is translucent and only its far half is drawn, behind the skeleton, so that the torque colours stay readable (display choice).
- The camera is fixed in the SKEL world frame (y up). `view 0 deg` looks along -z from the +z side. `view 60 deg` is the same camera turned 60 degrees about the vertical axis towards +x. The views do not follow the body heading, so `view 0 deg` shows the front of the body only when the subject faces +z.
- Each view is centred on the bounding box of the skin and of the drawn arrows (CoP points and arrow tips), so the feet and the arrow bases stay in the frame. Both png panels use the same scale.
- The arrow starts at `cop_aligned_foot` and follows `grf_aligned` (SKEL world frame, y up). Its length is 0.9 mm per N. Forces below 10 N have no arrow. Each arrow is moved 8 cm towards the camera along the viewing axis, so it stays in front of the foot and is not shifted sideways.
- Arrow colours (display choices): a foot arrow is green where `grf_foot_ok` and `cop_frame_ok` hold for that foot and frame. The Falisse2017 chair channel (`contact_bodies` = `pelvis`) is blue where `grf_frame_ok` holds. Every other arrow is grey, because its force or its CoP is masked. Masked values are not NaN, so the renderer applies these masks itself. It only changes the colour and removes nothing.
- The skeleton is coloured by `tau_selfconsistent`: the torque magnitude of each joint, ranked against the joint torques of the drawn frames and mapped to the plasma colour map (display only). Foot joint torques (ankle, subtalar, mtp) are doubled before the ranking, and a foot, forearm or wrist joint with zero torque takes the torque of its parent (display choices). The pelvis is grey, because its entry in `tau_selfconsistent` is the root residual, which mixes N m and N.
