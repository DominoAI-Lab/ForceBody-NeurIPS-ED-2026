"""Minimal SKEL rendering helpers for visualize.py (pyrender, headless EGL by default).

This module holds the parts of the authors' rendering utilities that visualize.py uses: the SKEL forward pass, the
camera, the pyrender scene, the GRF arrows and the torque colouring of the skeleton mesh through the SKEL skinning
weights. It needs the public SKEL package (https://github.com/MarilynKeller/SKEL) and the SKEL model files
(skel_male.pkl, skel_female.pkl, version 1.1.1) from https://skel.is.tue.mpg.de/ under the SKEL license.

Display frame: vertices are centred on the camera target and y is flipped, then each mesh is turned by 180 deg about x
in render_scene (pyrender camera convention). Positions passed to make_grf_arrow are in this display frame.
"""
import os
import pickle

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import pyrender  # noqa: E402

SKEL_REPO = 'https://github.com/MarilynKeller/SKEL'
SKEL_SITE = 'https://skel.is.tue.mpg.de/'

# AddB joint name -> DOF start, DOF count in the 37-DOF torque vector (tau_selfconsistent)
JOINT_DOF_MAPPING = [
    ("ground_pelvis", 0, 6), ("hip_r", 6, 3), ("walker_knee_r", 9, 1),
    ("ankle_r", 10, 1), ("subtalar_r", 11, 1), ("mtp_r", 12, 1),
    ("hip_l", 13, 3), ("walker_knee_l", 16, 1), ("ankle_l", 17, 1),
    ("subtalar_l", 18, 1), ("mtp_l", 19, 1), ("back", 20, 3),
    ("acromial_r", 23, 3), ("elbow_r", 26, 1), ("radioulnar_r", 27, 1),
    ("radius_hand_r", 28, 2), ("acromial_l", 30, 3), ("elbow_l", 33, 1),
    ("radioulnar_l", 34, 1), ("radius_hand_l", 35, 2),
]

# AddB joint -> SKEL joint indices that take its torque colour
ADDB_TO_SKEL_JOINT_MAP = {
    'ground_pelvis': [0], 'hip_r': [1], 'walker_knee_r': [2],
    'ankle_r': [3], 'subtalar_r': [4], 'mtp_r': [5],
    'hip_l': [6], 'walker_knee_l': [7], 'ankle_l': [8],
    'subtalar_l': [9], 'mtp_l': [10],
    'back': [11, 12, 13],
    'acromial_r': [14, 15], 'elbow_r': [16], 'radioulnar_r': [17], 'radius_hand_r': [18],
    'acromial_l': [19, 20], 'elbow_l': [21], 'radioulnar_l': [22], 'radius_hand_l': [23],
}

# A joint with zero torque takes the torque of its parent (display choice)
ADDB_PARENT_FALLBACK = {
    'mtp_r': 'subtalar_r', 'mtp_l': 'subtalar_l',
    'radioulnar_r': 'elbow_r', 'radioulnar_l': 'elbow_l',
    'radius_hand_r': 'elbow_r', 'radius_hand_l': 'elbow_l',
    'subtalar_r': 'ankle_r', 'subtalar_l': 'ankle_l',
    'ankle_r': 'walker_knee_r', 'ankle_l': 'walker_knee_l',
}

# Foot joints are drawn with a doubled torque magnitude (display choice)
ANKLE_BOOST_JOINTS = {'ankle_r', 'ankle_l', 'subtalar_r', 'subtalar_l', 'mtp_r', 'mtp_l'}
ANKLE_BOOST_FACTOR = 2.0

# Plasma colormap control points (5 points of matplotlib plasma)
_PLASMA_COLORS = [
    np.array([0.050383, 0.029803, 0.527975]),
    np.array([0.417642, 0.000564, 0.658390]),
    np.array([0.798216, 0.280197, 0.469538]),
    np.array([0.988362, 0.557937, 0.231441]),
    np.array([0.940015, 0.975158, 0.131326]),
]
_GRAY = np.array([0.85, 0.85, 0.85])

# view name -> rotation about the vertical axis in degrees (None = no rotation)
VIEWS = {'front': None, 'side60d': -60.0}


# ---------------------------------------------------------------------------
# SKEL model
# ---------------------------------------------------------------------------

def resolve_skel_model(path=None):
    """SKEL model folder: `path` (visualize.py --skel_model) or else the SKEL_MODEL_PATH environment variable."""
    path = path or os.environ.get('SKEL_MODEL_PATH')
    if not path:
        raise FileNotFoundError(
            'SKEL model folder not set. Pass --skel_model /path/to/skel_models or set SKEL_MODEL_PATH. The folder must '
            f'hold skel_male.pkl and skel_female.pkl (SKEL 1.1.1), downloaded from {SKEL_SITE}')
    path = os.path.expanduser(path)
    missing = [f for f in ('skel_male.pkl', 'skel_female.pkl') if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(f'SKEL model folder {path} does not contain {" and ".join(missing)}. '
                                f'Download the SKEL model files (version 1.1.1) from {SKEL_SITE}')
    return path


def import_skel():
    """The SKEL model class from the public SKEL package."""
    try:
        from skel.skel_model import SKEL
    except ImportError as e:
        raise ImportError(f'The SKEL package is not importable ({e}). Install it from {SKEL_REPO}: '
                          f'pip install git+{SKEL_REPO}.git, or clone it and run pip install -e . in the clone') from e
    return SKEL


def skel_meshes(betas, poses, trans, gender, model_dir, device='cpu', batch_size=64):
    """SKEL forward pass for the given frames.

    Args:
        betas [10], poses [N,46], trans [N,3]: SKEL parameters (SKEL world frame).
        gender: 'male' or 'female' (the trial's skel_gender).
        model_dir: folder with skel_<gender>.pkl.

    Returns:
        dict with skin_v [N,6890,3], skel_v [N,V_skel,3] (float32), skin_f, skel_f (face index arrays).
    """
    import torch
    SKEL = import_skel()
    model = SKEL(gender=gender, model_path=model_dir).to(device).eval()
    b = torch.from_numpy(np.asarray(betas)).float().to(device)
    p = torch.from_numpy(np.asarray(poses)).float().to(device)
    t = torch.from_numpy(np.asarray(trans)).float().to(device)
    skin_v, skel_v = [], []
    with torch.no_grad():
        for s in range(0, p.shape[0], batch_size):
            e = min(s + batch_size, p.shape[0])
            out = model(poses=p[s:e], betas=b.unsqueeze(0).expand(e - s, -1), trans=t[s:e], poses_type='skel',
                        skelmesh=True, dJ=None, pose_dep_bs=True)
            skin_v.append(out.skin_verts.cpu().numpy())
            skel_v.append(out.skel_verts.cpu().numpy())
    return {'skin_v': np.concatenate(skin_v), 'skel_v': np.concatenate(skel_v),
            'skin_f': model.skin_f.cpu().numpy(), 'skel_f': model.skel_f.cpu().numpy()}


def load_lbs_weights(model_dir, gender):
    """SKEL skeleton-mesh skinning weights [V_skel, 24] from skel_<gender>.pkl."""
    with open(os.path.join(model_dir, f'skel_{gender}.pkl'), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data['skel_weights'].toarray()


# ---------------------------------------------------------------------------
# Torque colouring
# ---------------------------------------------------------------------------

def plasma_color(t):
    """Map t in [0,1] to plasma RGB [3]."""
    t = np.clip(t, 0, 1)
    idx = t * 4
    i = int(idx)
    if i >= 4:
        return _PLASMA_COLORS[4].copy()
    frac = idx - i
    return _PLASMA_COLORS[i] * (1 - frac) + _PLASMA_COLORS[i + 1] * frac


def build_torque_percentiles(tau_all, frame_indices):
    """Sorted log10 joint-torque magnitudes over the given frames (pelvis residual excluded)."""
    all_torques = []
    for fi in frame_indices:
        tau = tau_all[min(fi, len(tau_all) - 1)]
        for jn, ds, nd in JOINT_DOF_MAPPING:
            if jn == 'ground_pelvis':
                continue
            mag = float(np.linalg.norm(tau[ds:ds + nd]))
            if mag > 0.05:
                all_torques.append(mag)
    if not all_torques:
        return np.array([])
    return np.sort(np.log10(np.array(all_torques)))


def compute_lbs_torque_colors(tau_frame, lbs_weights, torque_percentiles=None):
    """Per-vertex RGB [V,3] in [0,1]: per-joint torque magnitude -> plasma colour, blended by the skinning weights.

    With torque_percentiles (from build_torque_percentiles) the colour is the squared percentile rank of log10 |tau|.
    Without them it is log10 |tau| over the frame maximum.
    """
    skel_idx_to_torque = np.zeros(24)
    joint_torques = {}
    for jn, ds, nd in JOINT_DOF_MAPPING:
        mag = float(np.linalg.norm(tau_frame[ds:ds + nd]))
        joint_torques[jn] = mag
        if jn not in ADDB_TO_SKEL_JOINT_MAP:
            continue
        for si in ADDB_TO_SKEL_JOINT_MAP[jn]:
            if mag > skel_idx_to_torque[si]:
                skel_idx_to_torque[si] = mag

    for addb_name, parent_name in ADDB_PARENT_FALLBACK.items():
        if addb_name not in ADDB_TO_SKEL_JOINT_MAP:
            continue
        for si in ADDB_TO_SKEL_JOINT_MAP[addb_name]:
            if skel_idx_to_torque[si] == 0:
                pn = parent_name
                pt = joint_torques.get(pn, 0)
                while pt == 0 and pn in ADDB_PARENT_FALLBACK:
                    pn = ADDB_PARENT_FALLBACK[pn]
                    pt = joint_torques.get(pn, 0)
                if pt > 0:
                    skel_idx_to_torque[si] = pt

    for jn in ANKLE_BOOST_JOINTS:
        if jn not in ADDB_TO_SKEL_JOINT_MAP:
            continue
        for si in ADDB_TO_SKEL_JOINT_MAP[jn]:
            skel_idx_to_torque[si] *= ANKLE_BOOST_FACTOR

    if torque_percentiles is not None and len(torque_percentiles) > 0:
        def _color(mag):
            if mag < 0.05:
                return _GRAY
            idx = np.searchsorted(torque_percentiles, np.log10(mag))
            return plasma_color((idx / len(torque_percentiles)) ** 2.0)
    else:
        log_max = np.log10(max(skel_idx_to_torque.max(), 1.0))

        def _color(mag):
            if mag < 0.05:
                return _GRAY
            return plasma_color(np.clip(np.log10(mag) / log_max, 0, 1))

    joint_colors = np.array([_color(t) for t in skel_idx_to_torque])
    return np.clip(lbs_weights @ joint_colors, 0, 1)


# ---------------------------------------------------------------------------
# Camera, geometry, scene
# ---------------------------------------------------------------------------

def compute_camera(skel_data, frame_indices, size=768, focal=5000):
    """Camera that frames the skin vertices of all given frames.

    Returns:
        cam_t [3] camera translation, K4 [fx, fy, cx, cy], global_center [3] (centre of the skin vertices).
    """
    all_skin = np.concatenate([skel_data['skin_v'][fi] for fi in frame_indices], axis=0)
    gc = all_skin.mean(axis=0)
    ac = all_skin - gc
    ac[:, 1] *= -1
    y_extent = ac[:, 1].max() - ac[:, 1].min()
    x_extent = ac[:, 0].max() - ac[:, 0].min()
    tz = focal * max(y_extent, x_extent) / (size * 0.80)
    cam_t = np.array([0.0, 0.0, tz])
    K4 = [focal, focal, size // 2, size // 2]
    return cam_t, K4, gc


def prep_vertices(v, global_center):
    """Centre vertices [N,3] on global_center and flip y (display frame)."""
    out = v - global_center
    out[:, 1] *= -1
    return out


def make_grf_arrow(cop_pos, force_vec, scale=0.0009, z_offset=-0.08):
    """GRF arrow mesh (shaft + sphere head) from cop_pos along force_vec (display frame), or None below 10 N.

    The arrow length is |F| * scale (m per N) and the arrow is moved by z_offset towards the camera.
    """
    mag = np.linalg.norm(force_vec)
    if mag < 10:
        return None
    end = cop_pos + force_vec / mag * (mag * scale)
    offset = np.array([0.0, 0.0, z_offset])
    shaft = trimesh.creation.cylinder(radius=0.008, segment=np.array([cop_pos + offset, end + offset]))
    head = trimesh.creation.uv_sphere(radius=0.016, count=[8, 8])
    head.apply_translation(end + offset)
    return trimesh.util.concatenate([shaft, head])


def _materials():
    return {
        'skin_trans': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode='BLEND', baseColorFactor=(0.537, 0.765, 0.922, 0.55)),
        'grf': pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.9, alphaMode='OPAQUE', baseColorFactor=(0.15, 0.85, 0.15, 1.0)),
    }


def _raymond_lights():
    """Three directional lights."""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=0.75), matrix=matrix))
    return nodes


def render_scene(meshes_list, cam_t, K4, size, view='front'):
    """Render a list of (trimesh, pyrender material or None) on a white background. Returns [size,size,3] uint8."""
    bg_color = np.ones((size, size, 3), dtype=np.uint8) * 255
    renderer = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size, point_size=1.0)
    fx, fy, cx, cy = K4
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    for node in _raymond_lights():
        scene.add_node(node)
    deg = VIEWS[view]
    for idx, (tm, mat) in enumerate(meshes_list):
        m = tm.copy()
        if deg is not None:
            m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0]))
        m.apply_transform(trimesh.transformations.translation_matrix(cam_t))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))
        if mat is not None:
            scene.add(pyrender.Mesh.from_trimesh(m, material=mat), 'm%d' % idx)
        else:
            scene.add(pyrender.Mesh.from_trimesh(m), 'm%d' % idx)
    scene.add(camera, pose=np.eye(4))
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    mask = rgba.astype(np.float32)[:, :, [-1]] / 255.0
    final = rgba[:, :, :3] * mask + bg_color * (1 - mask)
    renderer.delete()
    return final.astype(np.uint8)


def render_panel_torque(skel_data, frame_indices, cam_t, K4, gc, size, view='front', lbs_weights=None,
                        torque_percentiles=None, show_grf=True, show_skin=True, extra_meshes=()):
    """Torque-coloured skeleton mesh, GRF arrows (green) and translucent skin for the given frames.

    skel_data keys: skel_v, skel_f, skin_v, skin_f (from skel_meshes), tau [N,37], grf_data (list per frame of
    (contact [C], grf [C,3], cop [C,3]) in the SKEL world frame). extra_meshes are (trimesh, material) pairs in the
    display frame, added after the body.
    """
    mat = _materials()
    tau_all = skel_data['tau']
    grf_data = skel_data.get('grf_data')
    meshes = []
    if show_grf and grf_data is not None:
        for fi in frame_indices:
            contact, grf, cop = grf_data[min(fi, len(grf_data) - 1)]
            for body_idx in range(len(contact)):
                if contact[body_idx] > 0 and body_idx < len(grf):
                    cop_centered = prep_vertices(cop[body_idx:body_idx + 1].copy(), gc)[0]
                    grf_vec = grf[body_idx].copy()
                    grf_vec[1] = -grf_vec[1]   # display y flip
                    arrow = make_grf_arrow(cop_centered, grf_vec)
                    if arrow is not None:
                        meshes.append((arrow, mat['grf']))
    for fi in frame_indices:
        sv = prep_vertices(skel_data['skel_v'][fi], gc)
        vc = compute_lbs_torque_colors(tau_all[min(fi, len(tau_all) - 1)], lbs_weights, torque_percentiles)
        colors = (np.column_stack([vc, np.ones(len(vc))]) * 255).astype(np.uint8)
        meshes.append((trimesh.Trimesh(sv, skel_data['skel_f'], vertex_colors=colors, process=False), None))
        if show_skin:
            skv = prep_vertices(skel_data['skin_v'][fi], gc)
            meshes.append((trimesh.Trimesh(skv, skel_data['skin_f'], process=False), mat['skin_trans']))
    return render_scene(meshes + list(extra_meshes), cam_t, K4, size, view)
