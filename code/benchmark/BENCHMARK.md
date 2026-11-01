# ForceBody force benchmark (spec 1.2.0, dataset 1.3.0)

Benchmark card for predicting joint torques (T-tau) and ground reaction forces (T-GRF) from motion on ForceBody 1.3.0.
The machine-readable definition is `benchmark/spec_v12.json` (spec version `forcebody-bench-1.2.0`, unchanged since
dataset 1.2.0). The folder `code/benchmark/` of the GitHub repository holds the card, the spec, a loader and cache
builder (`benchmark/build_bench_cache_v12.py`) and a reference scorer (`benchmark/score_v12.py`). The data are on
Hugging Face (https://huggingface.co/datasets/kwonjoon/ForceBody). Baselines retrained on dataset 1.2.1 are listed under
"Results".
Dataset 1.3.0 has the same benchmark inputs, labels, masks and splits as 1.2.1; only the 40 Hamner2013 trial keys
changed (`Hammer2013_*` -> `Hamner2013_*`, `uid_map_v12_to_v13.csv`), so prediction files keyed by 1.2.x `trial_uid`
must have these 40 keys renamed. The ForceBody paper reports three-seed means on all test strata, see "Paper numbers".

## Tasks

| | T-tau (joint torques) | T-GRF (ground reaction force) |
|---|---|---|
| inputs (required) | `poses` [T,46], `trans_vel` [T,3], `betas` [10], `mass_kg`, `height_m` | same |
| inputs (optional) | `q`, `qd`, `qdd` (= `q_kin`, `qd_kin`, `qdd_kin` for `pass0`), `joints`, `addb_joints` | same |
| label | `tau_selfconsistent / mass_kg_dynamics` (N m/kg) | `grf_aligned[:, 0:2] / (mass_kg_dynamics * 9.81)` (body weights) |
| label frame | `addb_lab`, OpenSim Rajagopal DOFs (`dof_names_opensim`) | `skel_world`, y up, gravity -y; contact 0 = right foot, 1 = left foot |
| scored channels | the 20 direct DOFs (`direct_dof_mask`): hip x3, knee, ankle, MTP per side; elbow, radioulnar, wrist x2 per side | 6 channels: right and left foot, x / y / z |
| frames scored | `eval_tau_frame_ok` and `eval_torque_trial_ok` | `eval_grf_frame_ok` and `eval_grf_trial_ok` (stance frames only on running, see below) |
| channel groups | `all` (20, headline), `all_no_wrist` (16), `lower` (12), `upper` (8), `upper_no_wrist` (4) | `all` (6, headline), `vertical` (2), `horizontal` (4), `sum_feet` (two-foot total, 3) |
| prediction file | `.npz`, one key per `trial_uid`, [T,37] N m/kg in `dof_names` order | `.npz`, one key per `trial_uid`, [T,2,3] BW |

- `trans_vel` is `numpy.gradient(trans, 1 / fps_true_grid)` inside the trial (m/s). Models may derive further features
  (for example pose velocities) from the inputs of the same trial.
- Mass (decision D21). Inputs use the recorded `mass_kg`. Labels are divided by `mass_kg_dynamics` (the AddB pass-2
  skeleton mass with the approved rescale already applied). Never multiply by `mass_dyn_scale` again.
- Gravity constant 9.81 m/s^2 is the constant of the v1.1/v1.2 force and QC code.
- Ground reaction moment (GRM). In ForceBody the GRM of a contact is `contact_torque` (`addb_lab`) /
  `contact_torque_aligned` (`skel_world`), the free moment in N m acting at the centre of pressure (`cop` /
  `cop_aligned_foot`). The resultant moment of a foot about a point p is `(cop - p) x grf + contact_torque`. GRM and CoP
  are not scored in this benchmark (Carter2023 running free moments are about five times those of Hamner2013). The Falisse2017 chair channel (contact 2) is not a target.
- The wrist DOFs are scored in `all` as in the paper; the `all_no_wrist` group is reported next to it.
- Scored frames and trials. `eval_tau_frame_ok` = `tau_frame_ok` and `kin_frame_ok`; `eval_grf_frame_ok` =
  `grf_frame_ok` and `kin_frame_ok` (pelvis-flip frames are not scored). The trial flags `eval_torque_trial_ok` /
  `eval_grf_trial_ok` exclude trials whose force labels are unusable (force/motion-capture desync), single-channel
  force plates (`qc_grf_single_channel`, one plate carries both feet), byte-identical duplicates (`qc_duplicate_drop`),
  the Han2023 trimmed copies of a re-exported capture (`qc_overlap_pair_drop`), static calibration, transition and pause
  segments, and trials whose dynamics mass disagrees with the GRF (`qc_mass_dyn_vs_grf_flag`); `eval_torque_trial_ok`
  also requires a root residual of at most 0.1 BW. The overlapping Tiziana2019 capture pairs share no GRF-valid frame
  and are kept for force evaluation (`qc_overlap_motion_drop` applies to motion evaluation only).
- `grf_valid_mask` covers the frames with a measured force. On Carter2023 running it excludes the flight frames, so
  T-GRF on running is scored on stance frames only.

## Pose source and leakage

Every trial is read with `forcebody_io.load_trial(npz, pose_source)`. `--pose_source` defaults to `pass0` (alias `kin`),
the SKEL fit to the AddB kinematics pass, which never saw the force plates. `pass2` (alias `dyn`) is the fit to the AddB
dynamics pass, which AddBiomechanics adjusted to agree with the measured GRF. It can be selected, but the cache meta and
every result json then carry `leaky_input: true` and `official_table_eligible: false`, and such results are **not**
listed in the tables below. Recovering the total GRF from the centre-of-mass acceleration alone, without learning, has a
median error of 2.56 N/kg on pass-1 kinematics and 0.20 N/kg on pass-2 kinematics (DETAILS.md on Hugging Face,
"Leakage policy").

The inputs are an allowlist: only the benchmark inputs of the table above are accepted, and any other requested input
raises `ValueError` in the loader (`build_bench_cache_v12.check_inputs`). The fields known to carry force information
are also refused by name (`forcebody_io.LEAKY_INPUT_FIELDS`): the dynamics-pass pose set (`poses_dyn`, `trans_dyn`,
`joints_dyn`, `addb_joints_dyn`, `q_gt`, `qd_gt`, `qdd_gt`), the labels (`tau_*`, `mu_tau`, `sigma_tau`, `percentiles`,
`grf*`, `cop*`, `contact_*`, `grf_valid_mask`) and fields derived from the GRF (`mass_kg_grf`, `mass_dyn_scale`,
`mass_kg_dynamics*`, `cadence_spm_grf`, `stride_time_s_med`, `duty_factor_*`, `support_side*`, `caption_support_side`,
`floor_height_m`, `speed_belt_grf_mps`, `mc_*`, `eval_*`, `t2m_*`, `s8b_*`, `qc_*`, the usability flags, `kin_frame_ok`,
`gait_events.csv`).

## Splits

Person-level splits (decision D26): `test` = `split_addb == "test"`, `val` = `split_addb_val == "val"` (persons drawn
from the `split_addb` train persons, about 10% per study; pathological persons and the heavy runner Carter2023_S20 are
not drawn), `train` = the remaining `split_addb` train persons. The manifest column `split_addb_val` holds the three
values. Select checkpoints and hyperparameters on `val` only and score `test` once.

Counts from the meta json files of `build_bench_cache_v12.py` (pose source pass0, dataset 1.2.1; identical for 1.3.0).
Scored frames and trials include the pathological stratum; "trials with labels" are the trials with the trial flag
(`eval_torque_trial_ok` / `eval_grf_trial_ok`).

| split | persons | trials | frames | T-tau trials with labels | T-tau scored frames | T-GRF trials with labels | T-GRF scored frames |
|---|---|---|---|---|---|---|---|
| train | 109 | 8,374 | 7,617,847 | 6,537 | 5,376,431 | 7,835 | 6,258,580 |
| val | 12 | 1,098 | 1,033,898 | 903 | 800,542 | 1,065 | 870,408 |
| test | 19 | 914 | 1,038,063 | 816 | 731,143 | 869 | 758,323 |

## Metrics

- Per channel, over the scored frames: RMSE and MAE of the per-frame label-unit error (N m/kg for T-tau, BW for T-GRF),
  and the Pearson correlation (PCC) over the pooled frames. A constant prediction on a channel whose label varies
  scores PCC 0. PCC is undefined (left out of the group mean, the count is reported) only when the label variance is 0.
  A variance counts as 0 when it is at most 1e-10 of the sum of squares (numerical tolerance, our choice).
- Group value = mean over the channels of the group (as in paper Table 4).
- Aggregations. `pooled` (headline): all scored frames of the healthy stratum. `study_macro`: per-study pooled value,
  then the unweighted mean over studies. `per_study` and `per_activity` (`activity_recovered`) pooled within each group.
  `pathological` (Fregly2012 TKA, Li2021 stroke; decision D18): reported separately, never part of the headline.
  `all_strata`: both strata pooled, for reference.
- Uncertainty. 95% percentile intervals from 1000 person-level bootstrap resamples (seed 0) for `pooled` and
  `study_macro` (persons resampled within each study). The number of resamples and the seed are our choice; no
  citation is claimed. A study with one test person has a zero-width interval.

## How to run

Run the commands from the `code/` folder of the repository. `/path/to/ForceBody` is the downloaded dataset folder.

```bash
# caches (default pose source pass0; one .npz + .json per split)
python benchmark/build_bench_cache_v12.py --release /path/to/ForceBody --out bench_cache --splits train val test
# score a prediction file on the test split
python benchmark/score_v12.py --task tau --cache bench_cache/bench_test_pass0.npz --pred my_tau_pred.npz --out tau_result.json
python benchmark/score_v12.py --task grf --cache bench_cache/bench_test_pass0.npz --pred my_grf_pred.npz --out grf_result.json
```

```python
import sys
sys.path.insert(0, "benchmark")            # run from code/
import build_bench_cache_v12 as B
c = B.load_cache("bench_cache/bench_train_pass0.npz")
x = B.input_matrix(c)                      # [N, 61] = poses, trans_vel, betas, mass_kg, height_m
y, m = c["tau_nmkg"], c["tau_mask"]        # labels [N, 37] and the scored-frame mask
for i, uid, sl in B.iter_trials(c): ...    # per-trial frame slices
```

`score_v12.table_row(result, name)` formats one row of the tables below and refuses leaky or non-test results.

## Results

Baselines: the paper model code in `code/baselines/` (Table 4 torque models, the Transformer + UW of Table 5, the Table 3
GRF MLP and GLN) retrained on dataset 1.2.1 with the pass-0 caches of `build_bench_cache_v12.py` (2026-10-29). Inputs are
benchmark inputs only. The torque models use `poses`, `trans_vel`, `betas`, `mass_kg` and `height_m`, with pose velocity
and acceleration derived inside the trial. The GRF models use `poses`, their second difference and `joints` relative to
the pelvis. Training on `train`, checkpoint selection on `val`, `test` scored once. Every value is the mean ± standard
deviation (ddof 1) over three seeds. Seed 1 is the original run (torque models unseeded, UW seed 0, GRF seed 20260427),
seeds 2 and 3 use `--seed 2` and `--seed 3` (`code/baselines/README.md`). Every seed comes from a `score_v12.py` result
json with `official_table_eligible == true` (`score_v12.table_row`).

Two aggregations are listed. `all_strata` pools the healthy and pathological test persons, as in the paper tables.
`pooled` is the healthy stratum, the headline of this card. For every seed, `score_v12.py` also reports `study_macro`,
`per_study`, `per_activity` and the 95% person-level bootstrap intervals.

T-tau, test split, group `all` (20 DOFs), N m/kg. `all_strata`: 19 persons, 816 trials, 731,143 scored frames.
`pooled`: 17 persons, 732 trials, 701,174 scored frames.

| model | all_strata RMSE | all_strata MAE | all_strata PCC | pooled RMSE | pooled MAE | pooled PCC |
|---|---|---|---|---|---|---|
| MLP | 0.148 ± 0.003 | 0.103 ± 0.002 | 0.757 ± 0.002 | 0.148 ± 0.003 | 0.103 ± 0.002 | 0.760 ± 0.002 |
| Conv1D | 0.129 ± 0.002 | 0.086 ± 0.001 | 0.802 ± 0.002 | 0.129 ± 0.002 | 0.086 ± 0.001 | 0.803 ± 0.002 |
| LSTM | 0.131 ± 0.002 | 0.088 ± 0.001 | 0.795 ± 0.003 | 0.132 ± 0.002 | 0.088 ± 0.001 | 0.797 ± 0.003 |
| GRU | 0.127 ± 0.001 | 0.085 ± 0.001 | 0.805 ± 0.003 | 0.128 ± 0.001 | 0.085 ± 0.001 | 0.807 ± 0.003 |
| Mamba2 | 0.125 ± 0.001 | 0.082 ± 0.001 | 0.813 ± 0.004 | 0.125 ± 0.001 | 0.082 ± 0.001 | 0.814 ± 0.004 |
| Transformer | 0.155 ± 0.009 | 0.107 ± 0.007 | 0.760 ± 0.010 | 0.155 ± 0.009 | 0.107 ± 0.007 | 0.763 ± 0.010 |
| Transformer + UW | 0.149 ± 0.007 | 0.101 ± 0.007 | 0.772 ± 0.012 | 0.149 ± 0.007 | 0.101 ± 0.007 | 0.775 ± 0.012 |

T-GRF, test split, group `all` (6 channels), BW. `all_strata`: 19 persons, 869 trials, 758,323 scored frames. `pooled`:
17 persons, 780 trials, 725,875 scored frames. The column "Table 3 GRF" is the paper metric on the `all_strata` frames:
the mean over scored frames of the sum over the two feet of the 3D error norm, in N/kg (`grf_paper_metric` in
`code/baselines/aggregate.py`).

| model | Table 3 GRF (N/kg) | all_strata RMSE | all_strata MAE | all_strata PCC | pooled RMSE | pooled MAE | pooled PCC |
|---|---|---|---|---|---|---|---|
| MLP | 1.165 ± 0.011 | 0.048 ± 0.000 | 0.028 ± 0.000 | 0.945 ± 0.001 | 0.048 ± 0.001 | 0.028 ± 0.000 | 0.946 ± 0.001 |
| GLN | 1.136 ± 0.040 | 0.050 ± 0.001 | 0.027 ± 0.001 | 0.944 ± 0.005 | 0.049 ± 0.002 | 0.026 ± 0.001 | 0.945 ± 0.006 |

Pathological stratum (reported separately, test split, 2 persons, group `all`, three seeds).

| model | task | trials | pooled RMSE | pooled MAE | pooled PCC |
|---|---|---|---|---|---|
| MLP | T-tau | 84 | 0.133 ± 0.010 | 0.106 ± 0.010 | 0.551 ± 0.010 |
| Conv1D | T-tau | 84 | 0.118 ± 0.006 | 0.090 ± 0.005 | 0.619 ± 0.030 |
| LSTM | T-tau | 84 | 0.124 ± 0.012 | 0.096 ± 0.011 | 0.617 ± 0.014 |
| GRU | T-tau | 84 | 0.116 ± 0.006 | 0.088 ± 0.006 | 0.644 ± 0.012 |
| Mamba2 | T-tau | 84 | 0.115 ± 0.006 | 0.086 ± 0.006 | 0.663 ± 0.014 |
| Transformer | T-tau | 84 | 0.133 ± 0.003 | 0.102 ± 0.001 | 0.514 ± 0.032 |
| Transformer + UW | T-tau | 84 | 0.132 ± 0.008 | 0.101 ± 0.006 | 0.559 ± 0.023 |
| MLP | T-GRF | 89 | 0.054 ± 0.001 | 0.038 ± 0.001 | 0.796 ± 0.010 |
| GLN | T-GRF | 89 | 0.063 ± 0.012 | 0.046 ± 0.009 | 0.735 ± 0.051 |

## Paper numbers

The camera-ready ForceBody paper reports the force benchmark in Tables 3 (ForceBody rows), 4, 5, 14 and 15. These
tables are the baselines of `code/baselines/` retrained on dataset 1.2.1, whose benchmark inputs, labels, masks and
splits are identical in 1.3.0. Checkpoints were selected on the 12 validation persons (1,098 trials), and the 19 test
persons (914 trials) were scored once. Each value is the mean and standard deviation over three seeds on the
`all_strata` aggregation. The `all_strata` columns of the T-tau table above equal the "All joints" row of paper Tables 4
and 5, and the "Table 3 GRF" column equals the ForceBody rows of paper Table 3.

- Table 3. The two AddBiomechanics rows are taken from the reviewed build of the dataset, so they serve as a reference
  rather than a matched comparison. They took AddB **dynamics-pass (pass 2)** kinematics as input, which were fitted
  with the force plates (force-label leakage). They also differ from the ForceBody rows in model implementation,
  optimizer, training subset and validation. The GRM column of Table 3 is not a target of this benchmark.
- The submitted version of the paper used **ForceBody 1.0.0 (reviewed build)** with the v1 torque labels, the P80 subset
  (SKEL fit error at most the 80th percentile), the validation set equal to the test set and one seed. Those numbers
  must not be mixed with results on this benchmark. The torque labels changed between versions: scoring the v1.1
  `tau_selfconsistent` against the v1 labels gives All-joint RMSE / MAE / PCC 0.287 / 0.087 / 0.721, the same order as
  the model errors.
- The reference scorer contains the formula of the submitted paper (`score_v12.paper_style_metrics`, `--mode paper`: N m
  errors over the GRF-valid frames divided by the mean body mass, PCC per DOF). On the test cache of the submitted paper
  it reproduces the scoring function of that paper package exactly. This check is not a benchmark result.

## Known limitations

- A trial with a true trial flag can still have no scored frame (for example all frames outside `eval_tau_frame_ok`);
  such trials are skipped and counted in `n_trials_without_scored_frames`.
- CoP, GRM and the uncertainty fields (`mu_tau`, `sigma_tau`) are not benchmark targets.
