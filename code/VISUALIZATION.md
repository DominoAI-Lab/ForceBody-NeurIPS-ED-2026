# Visualization

`visualize.py` renders the SKEL body of one trial with GRF arrows at the centre of pressure.

```bash
python visualize.py --npz trials/Moore2015/S10/Moore2015_S10_walk_014.npz --mode png --pose_source pass0
python visualize.py --npz trials/Moore2015/S10/Moore2015_S10_walk_014.npz --mode mp4 --pose_source pass2
```

- `--pose_source pass0` (default, alias `kin`) draws `poses` / `trans`; `pass2` (alias `dyn`) draws `poses_dyn` / `trans_dyn`. The trial is read with `forcebody_io.load_trial`, so only one pose set is used.
- The body uses `skel_gender` and `betas`; the arrow starts at `cop_aligned_foot` and follows `grf_aligned` (SKEL world frame, y up).
- The skeleton is coloured by `tau_selfconsistent`. Foot arrows are green; the Falisse2017 chair channel (`contact_bodies` = `pelvis`) is drawn in blue (display choice).
- Requirements: the addb2skel package (installed, or its folder in `ADDB2SKEL_ROOT`), an EGL renderer and the SKEL model files.
