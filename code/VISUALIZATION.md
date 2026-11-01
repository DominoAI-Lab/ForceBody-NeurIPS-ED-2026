# Visualization

`visualize.py` renders the SKEL body of one trial with GRF arrows at the centre of pressure. It needs `forcebody_io.py` and `skel_render.py` from this folder, the public SKEL package and the SKEL model files.

## Install

1. Install the Python packages from this folder.

   ```bash
   pip install -r requirements_visualize.txt
   ```

   The file lists numpy, scipy, torch, pyrender, trimesh, PyOpenGL, imageio with ffmpeg, Pillow and the SKEL package from https://github.com/MarilynKeller/SKEL (pinned to commit `c32cf16`, the SKEL version used to build ForceBody). `visualize.py` uses only the SKEL model loader, which imports torch, numpy and scipy. If the other SKEL dependencies do not install, install SKEL alone with `pip install --no-deps "git+https://github.com/MarilynKeller/SKEL.git@c32cf16581295bff19399379efe5b776d707cd95"`.

2. Download the SKEL model files. Register at https://skel.is.tue.mpg.de/ and download the SKEL model (version 1.1.1, `skel_male.pkl` and `skel_female.pkl`) under the SKEL license. ForceBody does not redistribute them. The download is `skel_models_v1.1.zip`. It unpacks to a folder `skel_models_v1.1/` that holds the two files. Use that folder in step 3.

3. Give the model folder to `visualize.py` with `--skel_model /path/to/folder` or with the environment variable `SKEL_MODEL_PATH`. The flag takes precedence.

Rendering is offscreen with pyrender. `visualize.py` sets `PYOPENGL_PLATFORM=egl` unless the variable is already set, so it needs an EGL-capable driver (a display is not needed). The SKEL forward pass runs on the GPU when torch finds one and on the CPU otherwise, and only for the frames that are drawn.

Tested on Linux with EGL, Python 3.9, torch 2.8.0, pyrender 0.1.45, PyOpenGL 3.1.0, trimesh 3.23.5 and 4.12.2, imageio 2.37.2 with imageio-ffmpeg 0.6.0, Pillow 11.3.0, numpy 1.26.4, scipy 1.13.1 and SKEL commit `c32cf16`.

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
| `--fps` | 20 | mp4 frame rate |
| `--max-frames` | 150 | mp4 frame limit. A longer contact window is subsampled. |

## Outputs

- `png`: one 1120 x 592 image with two 560 x 560 panels (front view and a side view turned 60 degrees) under a title band. The frame is the single-support frame with the largest GRF (one foot above 100 N, the other below 10 N). Without such a frame it is the frame with the largest GRF on any channel.
- `mp4`: a 640 x 640 side view (60 degrees, H.264) over the contact window, from the first to the last frame with a GRF above 50 N on any channel, with at most `--max-frames` frames.
- The title gives the trial file, the activity, the pose source, the fit target (`skel_fit_target` or `skel_fit_target_dyn`), the frame index and the largest GRF of that frame in N.

## What is drawn

- `--pose_source pass0` (default, alias `kin`) draws `poses` / `trans`. `pass2` (alias `dyn`) draws `poses_dyn` / `trans_dyn`. The trial is read with `forcebody_io.load_trial`, so only one pose set is used.
- The body uses `skel_gender` and `betas`. The skin is translucent.
- The arrow starts at `cop_aligned_foot` and follows `grf_aligned` (SKEL world frame, y up). Its length is 0.9 mm per N. Forces below 10 N have no arrow.
- The skeleton is coloured by `tau_selfconsistent`: the torque magnitude of each joint, ranked against the joint torques of the drawn frames and mapped to the plasma colour map (display only). Foot joint torques (ankle, subtalar, mtp) are doubled before the ranking, and a foot, forearm or wrist joint with zero torque takes the torque of its parent (display choices).
- Foot arrows are green. The Falisse2017 chair channel (`contact_bodies` = `pelvis`) is drawn in blue (display choice).
