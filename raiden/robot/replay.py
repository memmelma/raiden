"""Replay recorded arm motion.

Two source formats are supported, selected automatically based on the contents
of the episode directory:

Raw (``robot_data.npz``)
    Joint commands are streamed directly at the original ~100 Hz timing,
    resampled to ``control_hz``.  No IK is required.

Processed (``lowdim/<frame>.pkl``)
    EE-pose action sequence at ~30 fps is loaded, IK is solved at each
    keyframe, and the result is interpolated to ``control_hz``.

    The action layout stored in lowdim is::

        [l_pos(3), l_rot9(9), l_gripper(1), r_pos(3), r_rot9(9), r_gripper(1)]

    Poses are in the left-arm base frame.  The right arm target is transformed
    back to right-arm base coordinates using the bimanual calibration before IK.
"""

import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from i2rt.robots.kinematics import Kinematics

from raiden._xml_paths import get_yam_4310_linear_xml_path
from raiden.robot.controller import RobotController, smooth_move_joints
from raiden.robot.replay_images import (
    LiveCameraGrabber,
    ProcessedFrameReader,
    RawFrameReader,
    list_recording_cameras,
)
from raiden.robot.rollout_logger import DOF, RolloutLogger


def _noop_keep_mask(
    joints: np.ndarray,
    eps_joint: float,
    eps_gripper: float,
) -> np.ndarray:
    """Return bool mask (N,) where True = keep this frame.

    Frame *i* is a no-op if the delta to frame *i+1* is below threshold for
    every channel.  The last frame is always kept so the trajectory end-pose is
    preserved.

    Args:
        joints: ``(N, 7)`` (single arm) or ``(N, 14)`` (bimanual) joint array.
        eps_joint: Max absolute delta (radians) for arm joints to count as a no-op.
        eps_gripper: Max absolute delta for the gripper channel to count as a no-op.
    """
    d = np.abs(np.diff(joints.astype(np.float64), axis=0))  # (N-1, D)
    dj_l = np.max(d[:, 0:6], axis=1)
    dg_l = d[:, 6]
    is_noop = (dj_l <= eps_joint) & (dg_l <= eps_gripper)
    if joints.shape[1] >= 14:
        dj_r = np.max(d[:, 7:13], axis=1)
        dg_r = d[:, 13]
        is_noop = is_noop & (dj_r <= eps_joint) & (dg_r <= eps_gripper)
    return np.concatenate([~is_noop, [True]])  # always keep last frame


def _load_raw_joints(
    ep_dir: Path,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load joint commands from a raw ``robot_data.npz`` file.

    Returns:
        joints_l: (N, 7) float32 — left arm commanded joint positions.
        joints_r: (N, 7) float32 or None — right arm commanded joint positions.
        timestamps: (N,) int64 nanoseconds.
    """
    npz_path = ep_dir / "robot_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No robot_data.npz in {ep_dir}")
    data = np.load(npz_path)
    joints_l = data["follower_l_joint_cmd"].astype(np.float32)
    joints_r_raw = data.get("follower_r_joint_cmd")
    joints_r = joints_r_raw.astype(np.float32) if joints_r_raw is not None else None
    timestamps = data["timestamps"].astype(np.int64)
    return joints_l, joints_r, timestamps


def _resample_joints(
    joints: np.ndarray,
    timestamps_ns: np.ndarray,
    control_hz: int,
) -> np.ndarray:
    """Resample a joint trajectory from variable-rate timestamps to ``control_hz``."""
    t_src = timestamps_ns / 1e9
    n_out = int(round((t_src[-1] - t_src[0]) * control_hz)) + 1
    t_fine = np.linspace(t_src[0], t_src[-1], n_out)
    return np.stack(
        [np.interp(t_fine, t_src, joints[:, d]) for d in range(joints.shape[1])],
        axis=1,
    ).astype(np.float32)


def _load_lowdim_pkls(ep_dir: Path) -> list[dict]:
    """Return all per-frame lowdim dicts from an episode directory."""
    lowdim_dir = ep_dir / "lowdim"
    pkl_files = sorted(lowdim_dir.glob("??????????.pkl"))
    if not pkl_files:
        pkl_files = sorted(lowdim_dir.glob("?????????.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No lowdim .pkl files found in {lowdim_dir}")
    frames = []
    for p in pkl_files:
        with open(p, "rb") as f:
            frames.append(pickle.load(f))
    return frames


def _load_joint_sequence(ep_dir: Path) -> Optional[np.ndarray]:
    """Return (N, 14) float32 commanded joint positions from lowdim pkls, or None.

    Layout: ``[l_arm(6), l_gripper(1), r_arm(6), r_gripper(1)]``.
    Returns None if ``action_joints`` is absent from the lowdim files.
    """
    frames = _load_lowdim_pkls(ep_dir)
    if "action_joints" not in frames[0]:
        return None
    return np.stack(
        [np.asarray(f["action_joints"], dtype=np.float32) for f in frames]
    )  # (N, 14)


def _load_action_sequence(ep_dir: Path) -> np.ndarray:
    """Return (N, 26) float32 array of EE poses loaded from lowdim pkls."""
    frames = _load_lowdim_pkls(ep_dir)
    if "action" not in frames[0]:
        raise KeyError(f"'action' key missing in lowdim files under {ep_dir}")
    return np.stack(
        [np.asarray(f["action"], dtype=np.float32) for f in frames]
    )  # (N, 26)


def _pose_from_action(action_vec: np.ndarray, offset: int) -> np.ndarray:
    """Extract a 4×4 TCP pose from a 26-D action vector.

    Args:
        action_vec: (26,) array.
        offset: 0 for left arm, 13 for right arm.
    """
    pos = action_vec[offset : offset + 3].astype(np.float64)
    rot = action_vec[offset + 3 : offset + 12].reshape(3, 3).astype(np.float64)
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = rot
    return T


def _upsample_joints(
    joint_seq: np.ndarray,
    use_right: bool,
    control_hz: int,
    camera_hz: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Linearly interpolate (N, 14) joint commands from camera_hz to control_hz.

    Returns:
        traj_l: (M, 7) left arm trajectory.
        traj_r: (M, 7) right arm trajectory or None.
    """
    n_keys = len(joint_seq)
    upsample = control_hz // camera_hz
    M = (n_keys - 1) * upsample + 1
    t_keys = np.arange(n_keys, dtype=np.float64)
    t_fine = np.linspace(0, n_keys - 1, M)

    def _interp(cols: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(t_fine, t_keys, cols[:, d]) for d in range(cols.shape[1])],
            axis=1,
        ).astype(np.float32)

    traj_l = _interp(joint_seq[:, :7])
    traj_r = _interp(joint_seq[:, 7:14]) if use_right else None
    return traj_l, traj_r


def _get_kinematics() -> Kinematics:
    return Kinematics(get_yam_4310_linear_xml_path(), "grasp_site")


def _fk_ee_xyz(kin: Kinematics, q6: np.ndarray) -> np.ndarray:
    """Return (3,) EE position from 6-DOF arm joints."""
    nq = kin._configuration.model.nq
    q = np.zeros(nq, dtype=np.float64)
    q[:6] = q6
    return kin.fk(q)[:3, 3].astype(np.float32)


def _ik_6dof(
    kin: Kinematics, target_pose: np.ndarray, init_q: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """IK wrapper that pads init_q to model nq and returns only the 6 arm joints."""
    nq = kin._configuration.model.nq
    if len(init_q) < nq:
        q_full = np.zeros(nq, dtype=np.float64)
        q_full[: len(init_q)] = init_q
        init_q = q_full
    success, q = kin.ik(target_pose, "grasp_site", init_q=init_q)
    return success, q[:6]


def _solve_ik_sequence(
    actions: np.ndarray,
    use_right: bool,
    robot_l_init: np.ndarray,
    robot_r_init: Optional[np.ndarray],
    control_hz: int,
    camera_hz: int = 30,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """IK all keyframes then upsample to control_hz.

    Action poses are stored per-arm in each arm's own base frame, so no
    coordinate transform is needed before IK.

    Returns:
        traj_l: (M, 7) float32 joint trajectory for left arm.
        traj_r: (M, 7) float32 or None if right arm not used.
    """
    n_keys = len(actions)
    upsample = control_hz // camera_hz

    print("Setting up i2rt IK...")
    kin = _get_kinematics()

    # Seed joint states from the current robot positions (6 arm joints).
    q_l = robot_l_init[:6].copy().astype(np.float64)
    q_r = (
        robot_r_init[:6].copy().astype(np.float64) if robot_r_init is not None else None
    )

    print(f"Solving IK for {n_keys} keyframes...")
    keys_l = np.zeros((n_keys, 6), dtype=np.float64)
    keys_r = np.zeros((n_keys, 6), dtype=np.float64) if use_right else None
    grips_l = actions[:, 12].astype(np.float32)
    grips_r = actions[:, 25].astype(np.float32) if use_right else None

    for i, act in enumerate(actions):
        T_tcp_l = _pose_from_action(act, offset=0)
        _, q_l = _ik_6dof(kin, T_tcp_l, q_l)
        keys_l[i] = q_l

        if use_right and q_r is not None:
            T_tcp_r = _pose_from_action(act, offset=13)
            _, q_r = _ik_6dof(kin, T_tcp_r, q_r)
            keys_r[i] = q_r

        if (i + 1) % 30 == 0 or i == n_keys - 1:
            print(f"  IK {i + 1}/{n_keys}", end="\r", flush=True)
    print()

    # Append gripper channel to arm joints.
    def _to_mujoco_7d(keys: np.ndarray, grips: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [keys.astype(np.float32), grips[:, None]], axis=1
        )  # (N, 7)

    traj_l_keys = _to_mujoco_7d(keys_l, grips_l)
    traj_r_keys = (
        _to_mujoco_7d(keys_r, grips_r) if use_right and keys_r is not None else None
    )

    # Linear interpolation from camera_hz to control_hz.
    M = (n_keys - 1) * upsample + 1
    t_keys = np.arange(n_keys, dtype=np.float64)
    t_fine = np.linspace(0, n_keys - 1, M)

    def _interp(traj: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(t_fine, t_keys, traj[:, d]) for d in range(traj.shape[1])],
            axis=1,
        ).astype(np.float32)

    traj_l = _interp(traj_l_keys)
    traj_r = _interp(traj_r_keys) if traj_r_keys is not None else None

    return traj_l, traj_r


def run_replay(
    recording_dir: Path,
    arms: str = "bimanual",
    speed: float = 1.0,
    control_hz: int = 150,
    camera_hz: int = 30,
    stride: int = 1,
    visualize: bool = False,
    filter_noop: bool = False,
    noop_eps_joint: float = 1e-3,
    noop_eps_gripper: float = 1e-3,
    noop_min_frames: int = 10,
    log_dir: Optional[str] = None,
    log_images: bool = False,
    image_log_hz: float = 2.0,
) -> None:
    """Replay a recorded episode.

    Auto-detects the source format:

    - **Raw** (``robot_data.npz`` present): streams joint commands directly,
      no IK required.
    - **Processed** (``lowdim/`` present): solves IK from EE poses.

    Args:
        recording_dir: Path to a raw or converted episode directory.
        arms: ``"bimanual"`` or ``"single"`` (left only).
        speed: Playback speed multiplier.  1.0 = real-time.
        control_hz: Command rate for streaming (default 150 Hz).
        camera_hz: Frame rate of the lowdim action sequence (default 30 Hz,
            processed source only).
        stride: Subsample every N-th frame to match the shardify stride
            (default 1 = native rate, 3 = 10 Hz from 30 Hz recordings).
        filter_noop: Drop static (no-op) frames before replay.
        noop_eps_joint: Max absolute joint delta (rad) to classify as a no-op.
        noop_eps_gripper: Max absolute gripper delta to classify as a no-op.
        noop_min_frames: Abort replay if fewer frames remain after filtering.
        log_dir: Root directory for rollout debug logs (replayed trajectory vs.
            actual robot state), in the same format ``rd infer`` uses — pass to
            ``scripts/visualize_rollout.py`` to compare replay against
            inference. Default: ``/home/reward/Projects/logs/``.
        log_images: Also periodically log (1) a live snapshot from the real
            robot's cameras and (2) the corresponding recorded frame from the
            dataset being replayed, so you can visually compare them. Off by
            default (opens real cameras + decodes the recording's video —
            has a cost, so opt in only when actively debugging).
        image_log_hz: How often to capture the live/dataset image pair, in Hz
            (default 2.0). Independent of ``control_hz``.
    """
    if (recording_dir / "robot_data.npz").exists():
        _run_raw_replay(
            recording_dir,
            arms=arms,
            speed=speed,
            control_hz=control_hz,
            stride=stride,
            visualize=visualize,
            filter_noop=filter_noop,
            noop_eps_joint=noop_eps_joint,
            noop_eps_gripper=noop_eps_gripper,
            noop_min_frames=noop_min_frames,
            log_dir=log_dir,
            log_images=log_images,
            image_log_hz=image_log_hz,
        )
    elif (recording_dir / "lowdim").exists():
        _run_processed_replay(
            recording_dir,
            arms=arms,
            speed=speed,
            control_hz=control_hz,
            camera_hz=camera_hz,
            stride=stride,
            visualize=visualize,
            filter_noop=filter_noop,
            noop_eps_joint=noop_eps_joint,
            noop_eps_gripper=noop_eps_gripper,
            noop_min_frames=noop_min_frames,
            log_dir=log_dir,
            log_images=log_images,
            image_log_hz=image_log_hz,
        )
    else:
        raise FileNotFoundError(
            f"Neither robot_data.npz nor lowdim/ found in {recording_dir}"
        )


def _run_raw_replay(
    recording_dir: Path,
    arms: str = "bimanual",
    speed: float = 1.0,
    control_hz: int = 150,
    stride: int = 1,
    visualize: bool = False,
    filter_noop: bool = False,
    noop_eps_joint: float = 1e-3,
    noop_eps_gripper: float = 1e-3,
    noop_min_frames: int = 10,
    log_dir: Optional[str] = None,
    log_images: bool = False,
    image_log_hz: float = 2.0,
) -> None:
    """Replay directly from raw joint commands in robot_data.npz (no IK)."""
    joints_l, joints_r, timestamps = _load_raw_joints(recording_dir)
    use_right = arms == "bimanual" and joints_r is not None

    if stride > 1:
        joints_l = joints_l[::stride]
        joints_r = joints_r[::stride] if joints_r is not None else None
        timestamps = timestamps[::stride]

    if filter_noop:
        combined = (
            np.concatenate([joints_l, joints_r], axis=1)
            if use_right and joints_r is not None
            else joints_l
        )
        keep = _noop_keep_mask(combined, noop_eps_joint, noop_eps_gripper)
        n_before = len(joints_l)
        joints_l = joints_l[keep]
        joints_r = joints_r[keep] if (use_right and joints_r is not None) else joints_r
        timestamps = timestamps[keep]
        n_after = len(joints_l)
        print(f"No-op filter: {n_before} → {n_after} frames ({n_before - n_after} dropped)")
        if n_after < noop_min_frames:
            raise RuntimeError(
                f"Only {n_after} frames remain after no-op filtering "
                f"(min={noop_min_frames}); aborting replay."
            )

    traj_l = _resample_joints(joints_l, timestamps, control_hz)
    traj_r = _resample_joints(joints_r, timestamps, control_hz) if use_right else None

    n_frames = len(traj_l)
    duration_s = (timestamps[-1] - timestamps[0]) / 1e9
    print(f"Recording : {recording_dir.name}  (raw)")
    print(f"Samples   : {len(joints_l)}  ({duration_s:.1f} s, stride={stride})")
    print(f"Control Hz: {control_hz} Hz  ({n_frames} frames after resample)")
    print(f"Arms      : {arms}")
    print(f"Speed     : {speed}x")

    dataset_reader = None
    if log_images:
        camera_names = list_recording_cameras(recording_dir)
        if camera_names:
            dataset_reader = RawFrameReader(
                recording_dir, camera_names, total_control_steps=n_frames
            )
        else:
            print(f"[replay_images] no cameras found under {recording_dir}/cameras/ — "
                  "dataset image logging disabled.")

    _stream_trajectories(
        traj_l,
        traj_r,
        use_right=use_right,
        speed=speed,
        control_hz=control_hz,
        visualize=visualize,
        recording_dir=recording_dir,
        log_dir=log_dir,
        log_images=log_images,
        image_log_hz=image_log_hz,
        dataset_reader=dataset_reader,
    )


def _run_processed_replay(
    recording_dir: Path,
    arms: str = "bimanual",
    speed: float = 1.0,
    control_hz: int = 150,
    camera_hz: int = 30,
    stride: int = 1,
    visualize: bool = False,
    filter_noop: bool = False,
    noop_eps_joint: float = 1e-3,
    noop_eps_gripper: float = 1e-3,
    noop_min_frames: int = 10,
    log_dir: Optional[str] = None,
    log_images: bool = False,
    image_log_hz: float = 2.0,
) -> None:
    """Replay from processed lowdim pkl files using IK from EE poses."""
    use_right = arms == "bimanual"

    actions = _load_action_sequence(recording_dir)
    if stride > 1:
        actions = actions[::stride]

    if filter_noop:
        joint_seq = _load_joint_sequence(recording_dir)
        if joint_seq is not None:
            if stride > 1:
                joint_seq = joint_seq[::stride]
            keep = _noop_keep_mask(joint_seq, noop_eps_joint, noop_eps_gripper)
            n_before = len(actions)
            actions = actions[keep]
            n_after = len(actions)
            print(
                f"No-op filter: {n_before} → {n_after} frames ({n_before - n_after} dropped)"
            )
            if n_after < noop_min_frames:
                raise RuntimeError(
                    f"Only {n_after} frames remain after no-op filtering "
                    f"(min={noop_min_frames}); aborting replay."
                )
        else:
            print(
                "No-op filter: skipped (no action_joints in lowdim; "
                "filtering requires processed data with joint annotations)"
            )

    effective_hz = camera_hz // stride
    n_keys = len(actions)
    duration_s = n_keys / effective_hz
    print(f"Recording : {recording_dir.name}  (processed, IK)")
    print(
        f"Keyframes : {n_keys}  ({duration_s:.1f} s at {effective_hz} fps, stride={stride})"
    )
    print(f"Control Hz: {control_hz} Hz  (upsample ×{control_hz // effective_hz})")
    print(f"Arms      : {arms}")
    print(f"Speed     : {speed}x")

    upsample = control_hz // effective_hz
    dataset_reader = None
    if log_images:
        camera_names = list_recording_cameras(recording_dir)
        if camera_names:
            dataset_reader = ProcessedFrameReader(
                recording_dir, camera_names, upsample=upsample, stride=stride
            )
        else:
            print(f"[replay_images] no cameras found under {recording_dir}/rgb/ — "
                  "dataset image logging disabled.")

    robot = RobotController(
        use_right_follower=use_right,
        use_left_follower=True,
        use_right_leader=False,
        use_left_leader=False,
    )
    robot.initialize_robots()

    try:
        robot_l_init = (
            robot.follower_l.get_joint_pos() if robot.follower_l else np.zeros(7)
        )
        robot_r_init = (
            robot.follower_r.get_joint_pos()
            if (use_right and robot.follower_r)
            else None
        )

        traj_l, traj_r = _solve_ik_sequence(
            actions,
            use_right=use_right and robot.follower_r is not None,
            robot_l_init=robot_l_init,
            robot_r_init=robot_r_init,
            control_hz=control_hz,
            camera_hz=effective_hz,
        )

        _stream_trajectories(
            traj_l,
            traj_r,
            use_right=use_right,
            speed=speed,
            control_hz=control_hz,
            robot=robot,
            visualize=visualize,
            recording_dir=recording_dir,
            log_dir=log_dir,
            log_images=log_images,
            image_log_hz=image_log_hz,
            dataset_reader=dataset_reader,
        )
    except KeyboardInterrupt:
        print("\nReplay interrupted.")
    finally:
        robot.move_to_home_positions()
        robot.close()


def _stream_trajectories(
    traj_l: np.ndarray,
    traj_r: Optional[np.ndarray],
    use_right: bool,
    speed: float,
    control_hz: int,
    recording_dir: Optional[Path] = None,
    robot: Optional["RobotController"] = None,
    visualize: bool = False,
    log_dir: Optional[str] = None,
    log_images: bool = False,
    image_log_hz: float = 2.0,
    dataset_reader: Optional[object] = None,
) -> None:
    """Connect to (or reuse) the robot, move to start, then stream joint trajectories.

    If *robot* is None a new ``RobotController`` is created and closed when
    done.  Pass an already-initialized controller to avoid reconnecting.

    Logs the replayed (commanded) trajectory alongside the actual robot state
    to ``log_dir`` (default ``/home/reward/Projects/logs/``) in the same
    format ``rd infer`` uses, so ``scripts/visualize_rollout.py`` can compare
    a ground-truth replay against a live-policy rollout of the same task.

    When ``log_images`` is set, also periodically saves (1) a live snapshot
    from the real robot's cameras and (2) the matching recorded frame from
    ``dataset_reader`` (a :class:`~raiden.robot.replay_images.ProcessedFrameReader`
    or :class:`~raiden.robot.replay_images.RawFrameReader`), at ``image_log_hz``.
    """
    owns_robot = robot is None
    if owns_robot:
        robot = RobotController(
            use_right_follower=use_right,
            use_left_follower=True,
            use_right_leader=False,
            use_left_leader=False,
        )
        robot.initialize_robots()

    run_name = "replay_" + (recording_dir.name if recording_dir else "unknown")
    run_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_kwargs = {"log_root": Path(log_dir)} if log_dir else {}
    rollout_logger = RolloutLogger(
        run_name=run_name,
        source="replay",
        extra_meta={
            # Resolve to absolute so offline backfill / viz can find bags
            # even if the shell cwd differs from the original replay cwd.
            "recording_dir": str(recording_dir.resolve()) if recording_dir else None,
            "log_images_enabled": log_images,
        },
        **logger_kwargs,
    )

    live_grabber: Optional[LiveCameraGrabber] = None
    image_log_every = max(1, round(control_hz / image_log_hz)) if image_log_hz > 0 else None
    if log_images:
        camera_names = list_recording_cameras(recording_dir) if recording_dir else []
        if camera_names:
            live_grabber = LiveCameraGrabber(camera_names)
        else:
            print("[replay_images] no recording cameras identified — live image logging disabled.")

    # ── Rerun setup ────────────────────────────────────────────────────────
    rr_kin = None
    cmd_pts_l: list = []
    cmd_pts_r: list = []
    act_pts_l: list = []
    act_pts_r: list = []
    log_every = max(1, control_hz // 30)  # downsample to ~30 Hz for Rerun

    if visualize:
        from urllib.parse import quote

        import rerun as rr

        rr.init("raiden_replay")
        grpc_port = 9878
        web_port = 9877
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False)
        viewer_url = f"http://localhost:{web_port}?url={quote(server_uri, safe='')}"
        print(f"\nRerun viewer:  {viewer_url}")
        print(
            f"SSH tunnel:    ssh -L {web_port}:localhost:{web_port}"
            f" -L {grpc_port}:localhost:{grpc_port} <host>\n"
        )
        rr_kin = _get_kinematics()

    n_frames = len(traj_l)
    try:
        # ── move to start pose ─────────────────────────────────────────────
        print("\nMoving to start position...")
        threads = []
        if robot.follower_l is not None:
            threads.append(
                threading.Thread(
                    target=smooth_move_joints,
                    args=(robot.follower_l, traj_l[0]),
                    kwargs={"time_interval_s": 3.0, "steps": 100},
                )
            )
        if use_right and robot.follower_r is not None and traj_r is not None:
            threads.append(
                threading.Thread(
                    target=smooth_move_joints,
                    args=(robot.follower_r, traj_r[0]),
                    kwargs={"time_interval_s": 3.0, "steps": 100},
                )
            )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # ── stream at control_hz ───────────────────────────────────────────
        print("Replaying... (Ctrl-C to stop)\n")
        dt_s = 1.0 / (control_hz * speed)
        t_start = time.monotonic()

        for i in range(n_frames):
            t_target = t_start + i * dt_s
            sleep = t_target - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

            if robot.follower_l is not None:
                robot.follower_l.command_joint_pos(traj_l[i])
            if use_right and robot.follower_r is not None and traj_r is not None:
                robot.follower_r.command_joint_pos(traj_r[i])

            # Log replayed (commanded) trajectory vs. actual robot state at
            # ~30 Hz (same cadence as the Rerun stream) so debug logs stay a
            # manageable size for long/fast (150 Hz) replays.
            if i % log_every == 0:
                right_cmd = (
                    traj_r[i]
                    if (use_right and traj_r is not None)
                    else np.zeros(DOF, dtype=np.float32)
                )
                left_cmd = traj_l[i]
                action_14d = np.concatenate([right_cmd, left_cmd]).astype(np.float32)

                right_state = (
                    robot.follower_r.get_joint_pos().astype(np.float32)
                    if (use_right and robot.follower_r is not None)
                    else np.zeros(DOF, dtype=np.float32)
                )
                left_state = (
                    robot.follower_l.get_joint_pos().astype(np.float32)
                    if robot.follower_l is not None
                    else np.zeros(DOF, dtype=np.float32)
                )
                state_14d = np.concatenate([right_state, left_state])

                # No safety-clip step during replay, so predicted == commanded.
                rollout_logger.log_step(
                    state_14d=state_14d,
                    action_pred_14d=action_14d,
                    action_cmd_14d=action_14d,
                )

            # Periodically log a live robot snapshot + the matching recorded
            # dataset frame, at image_log_hz (independent of log_every above).
            if log_images and image_log_every is not None and i % image_log_every == 0:
                if live_grabber is not None:
                    live_images = live_grabber.grab_all()
                    if live_images:
                        rollout_logger.log_images(i, live_images, group="live")
                if dataset_reader is not None:
                    dataset_images = dataset_reader.get_frame(i)
                    if dataset_images:
                        rollout_logger.log_images(i, dataset_images, group="dataset")

            if visualize and rr_kin is not None and i % log_every == 0:
                import rerun as rr

                rr.set_time("frame", sequence=i)

                cmd_pts_l.append(_fk_ee_xyz(rr_kin, traj_l[i, :6]))
                rr.log(
                    "trajectory/commanded/left",
                    rr.LineStrips3D([cmd_pts_l], colors=[[0, 220, 0]]),
                )

                if use_right and traj_r is not None:
                    cmd_pts_r.append(_fk_ee_xyz(rr_kin, traj_r[i, :6]))
                    rr.log(
                        "trajectory/commanded/right",
                        rr.LineStrips3D([cmd_pts_r], colors=[[0, 120, 255]]),
                    )

                if robot is not None:
                    if robot.follower_l is not None:
                        act_pts_l.append(
                            _fk_ee_xyz(rr_kin, robot.follower_l.get_joint_pos()[:6])
                        )
                        rr.log(
                            "trajectory/actual/left",
                            rr.LineStrips3D([act_pts_l], colors=[[255, 80, 0]]),
                        )
                    if use_right and robot.follower_r is not None:
                        act_pts_r.append(
                            _fk_ee_xyz(rr_kin, robot.follower_r.get_joint_pos()[:6])
                        )
                        rr.log(
                            "trajectory/actual/right",
                            rr.LineStrips3D([act_pts_r], colors=[[255, 0, 150]]),
                        )

            if i % 150 == 0:
                elapsed = time.monotonic() - t_start
                total_s = n_frames * dt_s
                print(
                    f"  {i}/{n_frames} frames  "
                    f"({elapsed:.1f} s elapsed / {total_s:.1f} s total)",
                    end="\r",
                )

        print(f"\nReplay complete ({n_frames} frames at {control_hz} Hz).")

    except KeyboardInterrupt:
        print("\nReplay interrupted.")
    finally:
        rollout_logger.save()
        if live_grabber is not None:
            live_grabber.close()
        if dataset_reader is not None:
            dataset_reader.close()
        if owns_robot:
            robot.move_to_home_positions()
            robot.close()
