"""ForceBody trial loader (release 1.3.0; trial format v1.2) with an explicit pose source (numpy only).

ForceBody (since v1.2) ships two SKEL pose sets per trial (see README, "Choosing a pose source"):

  pose_source  alias  fields used                                            fitted to
  'pass0'      'kin'  poses, trans, joints, q_kin, qd_kin, qdd_kin,          AddB kinematics pass (pass 0 + 30 Hz
                      addb_joints_kin                                        filter = pass 1); never saw the force plates
  'pass2'      'dyn'  poses_dyn, trans_dyn, joints_dyn, q_gt, qd_gt, qdd_gt,  AddB dynamics pass (pass 2); adjusted by
                      addb_joints_dyn                                        AddBiomechanics to agree with the measured GRF

Default use: force / torque estimation and any model or metric that takes motion as input and predicts or scores forces
-> 'pass0'. Text-to-motion (motion only, or motion + GRF + torque) -> 'pass2'. Feeding 'pass2' kinematics to a model
that predicts or scores forces leaks the force labels into its input (README, "Leakage policy").

    import forcebody_io as fio
    d = fio.load_trial('trials/Moore2015/S10/Moore2015_S10_walk_014.npz', pose_source='pass0')
    d['poses'], d['trans'], d['joints'], d['q'], d['pose_usable'], d['grf_aligned'], d['tau_selfconsistent'] ...

load_trial() returns a dict with the common keys
  poses [T,46], trans [T,3], joints [T,24,3]   SKEL pose, root translation and joint centres (SKEL world frame)
  q, qd, qdd [T,37]                            AddB joint angles / velocities / accelerations of the same pass
  addb_joints [T,20,3]                         AddB joint centres of the same pass (SKEL world frame)
  pose_usable (bool)                           trial-level: always True for 'pass0'; `poses_dyn_usable` for 'pass2'
  pose_source ('pass0' or 'pass2'), pose_fit_target (skel_fit_target or skel_fit_target_dyn), q_source (str)
  arm_event_ok [T,2], trunk_event_ok [T], fit_event_ok [T,7], addb_limit_mask [T,11],
  knee_clamp_frame [T,2] (1.2.1; right, left)
                                               pose masks of the chosen set (True = usable; README, "Pose masks of the
                                               fix rounds"); for 'pass2' also use dyn_root_jump_ok [T]
plus every other field of the trial file unchanged (GRF, CoP, torques, masks, metadata). The raw pose-set fields of
BOTH sources (poses, poses_dyn, trans, trans_dyn, joints, joints_dyn, q_kin*, q_gt*, addb_joints_kin, addb_joints_dyn)
are not copied, so a model sees only the source that was asked for; use numpy.load on the file for raw access.

Re-timed trials (skel_fit_target_dyn == 'kin_retimed'): the force stream was shifted in time to match the motion capture,
the pass-2 fit (bent to the misaligned force) is replaced by the pass-0 fit, and q_gt / qd_gt / qdd_gt / tau_gt_det are
not usable (q_gt_usable False). For 'pass2' on these trials q / qd / qdd therefore come from q_kin / qd_kin / qdd_kin and
q_source says so.
"""
import numpy as np

POSE_SOURCES = {'pass0': 'pass0', 'kin': 'pass0', 'pass2': 'pass2', 'dyn': 'pass2'}

# per canonical source: common key -> trial field
_FIELDS = {
    'pass0': {'poses': 'poses', 'trans': 'trans', 'joints': 'joints', 'q': 'q_kin', 'qd': 'qd_kin', 'qdd': 'qdd_kin',
              'addb_joints': 'addb_joints_kin', 'arm_event_ok': 'arm_event_ok_kin',
              'trunk_event_ok': 'trunk_event_ok_kin', 'fit_event_ok': 'fit_event_ok_kin',
              'addb_limit_mask': 'addb_limit_mask_kin', 'knee_clamp_frame': 'knee_clamp_frame_kin'},
    'pass2': {'poses': 'poses_dyn', 'trans': 'trans_dyn', 'joints': 'joints_dyn', 'q': 'q_gt', 'qd': 'qd_gt',
              'qdd': 'qdd_gt', 'addb_joints': 'addb_joints_dyn', 'arm_event_ok': 'arm_event_ok_dyn',
              'trunk_event_ok': 'trunk_event_ok_dyn', 'fit_event_ok': 'fit_event_ok_dyn',
              'addb_limit_mask': 'addb_limit_mask_dyn', 'knee_clamp_frame': 'knee_clamp_frame_dyn'},
}
_POSE_SET_KEYS = {'poses', 'trans', 'joints', 'poses_dyn', 'trans_dyn', 'joints_dyn', 'q_kin', 'qd_kin', 'qdd_kin',
                  'q_gt', 'qd_gt', 'qdd_gt', 'addb_joints_kin', 'addb_joints_dyn',
                  # v1.2 fix-round pose masks of each set (README "Pose masks of the fix rounds")
                  'arm_event_ok_kin', 'arm_event_ok_dyn', 'trunk_event_ok_kin', 'trunk_event_ok_dyn',
                  'fit_event_ok_kin', 'fit_event_ok_dyn', 'addb_limit_mask_kin', 'addb_limit_mask_dyn',
                  'knee_clamp_frame_kin', 'knee_clamp_frame_dyn'}   # 1.2.1: SKEL knee on its 0 bound, AddB knee flexed

# The leakage rule is an ALLOWLIST (README, "Leakage policy"): a model that predicts or scores forces or torques may take
# as input only ALLOWED_INPUT_FIELDS (the pass-0 pose set through load_trial(..., 'pass0'), body shape and person
# descriptors) plus person-free text metadata. LEAKY_INPUT_FIELDS is a denylist safety net of fields known to carry
# information from the measured forces (the dynamics pass, force / torque labels and everything derived from the GRF).
# Prefix entries end with '*'.
ALLOWED_INPUT_FIELDS = ('poses', 'trans', 'joints', 'q_kin', 'qd_kin', 'qdd_kin', 'addb_joints_kin', 'betas',
                        'skel_gender', 'mass_kg', 'height_m', 'sex', 'age')
LEAKY_INPUT_FIELDS = ('poses_dyn', 'trans_dyn', 'joints_dyn', 'addb_joints_dyn', 'q_gt', 'qd_gt', 'qdd_gt', 'tau_*',
                      'mu_tau', 'sigma_tau', 'percentiles', 'grf*', 'cop*', 'contact_*', 'grf_valid_mask',
                      # S19 F8: GRF-derived fields
                      'mass_kg_grf', 'mass_dyn_scale', 'mass_kg_dynamics*', 'cadence_spm_grf', 'stride_time_s_med',
                      'duty_factor_*', 'support_side*', 'caption_support_side', 'floor_height_m', 'speed_belt_grf_mps',
                      'mc_*', 'eval_*', 't2m_*', 's8b_*', 'qc_*', 'force_label_usable*', 'torque_label_usable',
                      'poses_dyn_usable*', 'kin_frame_ok', 'gait_events',
                      # v1.2 fix rounds: every field of the dynamics-pass pose set (dyn_target_source follows the
                      # GRF-valid windows)
                      'dyn_*', '*_dyn')


def canonical_pose_source(pose_source):
    """'pass0'/'kin' -> 'pass0', 'pass2'/'dyn' -> 'pass2'; anything else (including None) raises ValueError."""
    if not isinstance(pose_source, str) or pose_source not in POSE_SOURCES:
        raise ValueError(f"pose_source must be one of {sorted(POSE_SOURCES)} (required, no default); got {pose_source!r}")
    return POSE_SOURCES[pose_source]


def is_leaky_field(name):
    """True if `name` is on the leakage list (exact name or prefix entry)."""
    for f in LEAKY_INPUT_FIELDS:
        if ((f.endswith('*') and name.startswith(f[:-1])) or (f.startswith('*') and name.endswith(f[1:]))
                or name == f):
            return True
    return False


def load_trial(npz_path, pose_source=None):
    """Load one ForceBody trial (v1.2 trial format, release 1.2.x or 1.3.0) with the pose set chosen by `pose_source` (required: 'pass0'|'kin'|'pass2'|'dyn')."""
    src = canonical_pose_source(pose_source)
    with np.load(npz_path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    out = {k: v for k, v in raw.items() if k not in _POSE_SET_KEYS}
    fmap = dict(_FIELDS[src])
    q_source = fmap['q']
    if src == 'pass2' and str(raw['skel_fit_target_dyn']) == 'kin_retimed':
        # q_gt (AddB pass 2) was fitted to the misaligned force; the re-timed labels use q_kin (README, re-timed trials)
        fmap.update(q='q_kin', qd='qd_kin', qdd='qdd_kin')
        q_source = 'q_kin (re-timed trial: q_gt is not usable)'
    for common, field in fmap.items():
        out[common] = raw[field]
    out['pose_source'] = np.array(src)
    out['pose_fit_target'] = raw['skel_fit_target'] if src == 'pass0' else raw['skel_fit_target_dyn']
    out['q_source'] = np.array(q_source)
    out['pose_usable'] = np.bool_(True) if src == 'pass0' else np.bool_(bool(raw['poses_dyn_usable']))
    return out
