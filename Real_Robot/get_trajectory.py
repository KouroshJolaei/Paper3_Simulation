# get_trajectory.py
import numpy as np
from scipy.spatial.transform import Rotation as SciR
def enforce_plane_constraint(joint_trajectory, fk_solver, joint_names, fixed_axis='fixed_axis', tolerance=0.005):
    """
    Keep EE within a 2D plane by fixing one axis (x, y, or z).
    Example: fixed_axis='y' => allows motion in XZ plane.
    """
    filtered_trajectory = []
    kept_count = 0

    for joint_pos in joint_trajectory:
        pose = fk_solver.get_fk(joint_names, joint_pos)
        if pose is None:
            continue

        axis_value = getattr(pose.position, fixed_axis)
        if abs(axis_value) < tolerance:
            filtered_trajectory.append(joint_pos)
            kept_count += 1

    print(f"🛠 Plane constraint kept {kept_count} / {len(joint_trajectory)}")
    return filtered_trajectory
def plan_trajectory_with_constraint(current_joints, target_joints, fk_solver=None, joint_names=None, plane_axis=None, lock_joint_index=None, lock_joint_value=None):
    """
    Plan a trajectory from current to target joints, with optional FK-based plane constraint and joint locking.
    """
    trajectory = generate_linear_trajectory(
        current_joints,
        target_joints,
        num_points=500,
        lock_joint_index=lock_joint_index
    )

    if fk_solver and plane_axis and joint_names:
        trajectory = enforce_plane_constraint(
            joint_trajectory=trajectory,
            fk_solver=fk_solver,
            joint_names=joint_names,
            fixed_axis=plane_axis
        )

    print(f"🔢 Trajectory length after interpolation and plane filtering: {len(trajectory)}")

    if len(trajectory) < 2:
        print("⚠️ Not enough points after filtering. Using unlocked fallback trajectory.")
        trajectory = generate_linear_trajectory(current_joints, target_joints)

    return trajectory
def project_point_to_plane(point, plane_point, plane_normal):
    """
    Projects a 3D point onto a plane defined by a point and a normal vector.
    """
    point = np.array(point)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    v = point - plane_point
    dist = np.dot(v, plane_normal)
    projected = point - dist * plane_normal
    return projected
def enforce_arbitrary_plane_constraint(joint_trajectory, fk_solver, joint_names, plane_point, plane_normal, tolerance='tolerance'):
    """
    Filters joint poses to only keep those whose EE position lies near a plane defined by arbitrary orientation.
    """
    filtered_trajectory = []
    kept_count = 0

    for joint_pos in joint_trajectory:
        pose = fk_solver.get_fk(joint_names, joint_pos)
        if pose is None:
            continue

        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        proj = project_point_to_plane(pos, plane_point, plane_normal)
        dist = np.linalg.norm(pos - proj)
        print(f"Distance to plane: {dist:.4f}")

        if dist < tolerance:
            filtered_trajectory.append(joint_pos)
            kept_count += 1

    print(f"🛠 Plane constraint (arbitrary) kept {kept_count} / {len(joint_trajectory)}")
    return filtered_trajectory
def plan_trajectory_in_finger_plane_via_jacobian(fk_solver, jacobian_func, joint_names, current_joints, finger_plane, dx_dy_final, steps=50):
    """
    Generate a joint-space trajectory by stepping in the finger plane and projecting via Jacobian.

    Args:
        fk_solver: Your FK client.
        jacobian_func: Callable that takes joint angles and returns the Jacobian (6xN).
        joint_names (list): Joint names.
        current_joints (list): Starting joint angles.
        finger_plane (dict): Dict with 'point' and 'normal'.
        dx_dy_final (tuple): Final delta in 2D finger plane (in meters).
        steps (int): Number of interpolation steps.

    Returns:
        List of joint states (trajectory).
    """
    q = np.array(current_joints)
    trajectory = [q.tolist()]

    normal = np.array(finger_plane["normal"])

    def orthonormal_basis(n):
        n = n / np.linalg.norm(n)
        other = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        v1 = np.cross(n, other)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(n, v1)
        return v1, v2

    v1, v2 = orthonormal_basis(normal)

    dx, dy = dx_dy_final
    total_disp = dx * v1 + dy * v2

    for alpha in np.linspace(0, 1, steps):
        disp = alpha * total_disp

        # FK to get EE frame
        pose = fk_solver.get_fk(joint_names, q.tolist())
        if pose is None:
            print("❌ FK failed at step", alpha)
            break

        current_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        target_pos = current_pos + disp

        delta_cartesian = target_pos - current_pos  # 3D delta
        delta_twist = np.concatenate((delta_cartesian, np.zeros(3)))  # No rotation


        # Jacobian step
        J = jacobian_func(q.tolist())  # must return 6xN matrix
        # Only use linear component
        delta_twist = delta_cartesian  # shape (3,)
        if J.shape[0] == 6:
            J = J[:3, :]  # keep only linear rows
        dq = np.linalg.pinv(J) @ delta_twist

        q = q + dq
        trajectory.append(q.tolist())

    print(f"✅ Trajectory created with {len(trajectory)} points via Jacobian projection.")
    return trajectory
def generate_linear_trajectory(current_joints, target_joints, num_points=100, lock_joint_index=None):
    current_joints = np.array(current_joints, dtype=float)
    target_joints  = np.array(target_joints,  dtype=float)
    traj = []
    for alpha in np.linspace(0.0, 1.0, num_points):
        q = (1.0 - alpha) * current_joints + alpha * target_joints
        if lock_joint_index is not None:
            q[lock_joint_index] = current_joints[lock_joint_index]
        traj.append(q.tolist())
    return traj

def _quat_to_R(q):
    # q: geometry_msgs/Quaternion-like (x,y,z,w)
    return SciR.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
def _rodrigues(axis, angle):
    ax = np.asarray(axis, float)
    ax /= (np.linalg.norm(ax) + 1e-12)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]], float)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
def _dls(J, v, lam=0.02):
    # damped least-squares solve for J dq ≈ v
    J = np.asarray(J, float)
    m, n = J.shape
    if m <= n:
        # dq = J^T (J J^T + λ^2 I)^{-1} v
        JT = J.T
        return JT @ np.linalg.solve(J @ JT + (lam**2) * np.eye(m), v)
    else:
        # overdetermined: dq = (J^T J + λ^2 I)^{-1} J^T v
        JT = J.T
        return np.linalg.solve(JT @ J + (lam**2) * np.eye(n), JT @ v)
def _adjoint_inv(R, p):
    # Ad_g^{-1} mapping spatial(base) twist -> body(tool) twist
    p = np.asarray(p, float).reshape(3)
    R = np.asarray(R, float).reshape(3, 3)
    px = np.array([[0, -p[2],  p[1]],
                   [p[2],  0, -p[0]],
                   [-p[1], p[0], 0]], float)
    Rt = R.T
    A11 = Rt
    A12 = -Rt @ px
    A21 = np.zeros((3, 3))
    A22 = Rt
    return np.block([[A11, A12],
                     [A21, A22]])
def plan_trajectory_on_finger_plane(
    current_joints,
    target_joints,
    fk_func,                  # q -> Pose (geometry_msgs/Pose)
    jacobian_func,            # returns 6xN (preferred) or 3xN (pos-only)
    finger_plane,             # {"point":[x,y,z], "normal":[nx,ny,nz]}
    pad_normal_axis='x',      # which tool axis is finger-pad normal ('x'|'y'|'z')
    theta_deg=0.0,            # in-plane rotation to apply over the path (deg)
    R_tool0_start=None,       # 3x3 start tool rotation (world); if None, read from FK
    lock_joint_index=None,    # optional: hold this joint constant during corrections
    joint_limits=None,        # optional: (lower, upper) arrays for clamping
    num_points=120,
    tol=0.005,                # plane distance tolerance (meters)
    max_corr_iters=2,         # iterations per corrected waypoint
    step_gain=0.4,            # position correction gain
    orient_gain=0.6,          # orientation (in-plane yaw) correction gain
    corr_stride=3,            # correct every Nth waypoint (and always last)
    jacobian_frame='spatial', # 'spatial' (base/world) or 'body' (tool)
    jacobian_order='lin_ang', # 'lin_ang' (v;w) or 'ang_lin' (w;v)
    dls_lambda=0.02,          # damping for DLS
    max_step_rad=0.05         # clamp |dq| per correction step (rad)
):
    # --- plane data ---
    p_plane = np.asarray(finger_plane["point"],  float).reshape(3)
    n_plane = np.asarray(finger_plane["normal"], float).reshape(3)
    n_plane /= (np.linalg.norm(n_plane) + 1e-12)

    # --- desired in-plane rotation path (about the pad-normal axis) ---
    theta_rad_total = np.deg2rad(theta_deg)
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[pad_normal_axis]

    # Get start tool rotation if not supplied
    if R_tool0_start is None:
        pose0 = fk_func(np.asarray(current_joints, float))
        R_tool0_start = _quat_to_R(pose0.orientation)
    else:
        R_tool0_start = np.asarray(R_tool0_start, float)

    # Fixed world axis for yaw = start tool’s pad-normal expressed in world
    axis_world0 = R_tool0_start[:, axis_idx]
    axis_world0 /= (np.linalg.norm(axis_world0) + 1e-12)

    # Sanity check: plane normal vs pad-normal alignment (informational)
    ang = np.degrees(np.arccos(np.clip(axis_world0 @ n_plane, -1.0, 1.0)))
    if ang > 5.0:
        print(f"⚠️ plane normal and pad-normal differ by {ang:.1f}° — check frames/axes.")

    # --- baseline joint interpolation (linear) ---
    cur = np.asarray(current_joints, float).reshape(-1)
    tgt = np.asarray(target_joints,  float).reshape(-1)
    N = len(cur)
    traj = []
    for a in np.linspace(0.0, 1.0, num_points):
        q = (1.0 - a) * cur + a * tgt
        if lock_joint_index is not None:
            q[lock_joint_index] = cur[lock_joint_index]
        traj.append(q.tolist())

    corrected = []
    for i, q in enumerate(traj):
        q = np.asarray(q, float)

        # Only correct every corr_stride-th point (and always the last)
        if (i % corr_stride) != 0 and i != (len(traj) - 1):
            corrected.append(q.tolist())
            continue

        # FK at waypoint
        pose = fk_func(q)
        p = np.array([pose.position.x, pose.position.y, pose.position.z], float)
        R_curr = _quat_to_R(pose.orientation)

        # --- (A) plane-distance correction (only along plane normal) ---
        err_n = float(n_plane.dot(p - p_plane))          # signed distance to plane
        pos_corr = -step_gain * err_n * n_plane          # move along plane normal

        # --- (B) in-plane yaw correction about axis_world0 ---
        s = i / max(1, (len(traj) - 1))                  # 0..1 progress
        theta_i = s * theta_rad_total
        R_des = _rodrigues(axis_world0, theta_i) @ R_tool0_start
        R_err = R_des @ R_curr.T
        omega_err = SciR.from_matrix(R_err).as_rotvec()  # world rotvec
        # Keep only component about axis_world0
        omega_plane = orient_gain * (axis_world0 * (omega_err @ axis_world0))

        # --- (C) Build twist and solve with J (DLS), with frame/order handling ---
        J = np.asarray(jacobian_func(q), float)
        if J.ndim != 2:
            raise RuntimeError("jacobian_func must return (M x N) array")

        # Lock the joint during corrections (zero out its column influence)
        if lock_joint_index is not None and 0 <= lock_joint_index < J.shape[1]:
            # Make sure dq doesn't move the locked joint
            # (we'll also zero dq[lock] right after solving as a safeguard)
            pass

        if J.shape[0] == 3:
            # Position-only Jacobian -> can keep plane, but won't track yaw
            twist_for_J = pos_corr
            dq = _dls(J, twist_for_J, lam=dls_lambda)
        else:
            # Expect 6xN
            if J.shape[0] != 6:
                raise RuntimeError(f"Unexpected Jacobian shape {J.shape}, expected 6xN or 3xN")

            # Spatial twist in world/base frame as [v; w]
            twist6 = np.r_[pos_corr, omega_plane]

            # If Jacobian is BODY/TOOL frame, transform twist to that frame at current pose
            if jacobian_frame.lower() == 'body':
                Ad_inv = _adjoint_inv(R_curr, p)  # spatial->body
                twist6 = Ad_inv @ twist6

            # Reorder rows if Jacobian order is [w; v]
            if jacobian_order.lower() == 'ang_lin':
                # J is [w; v]; reorder it to match twist [v; w] OR reorder twist to [w; v]
                # Easier: reorder twist into [w; v]
                twist6 = np.r_[twist6[3:], twist6[:3]]

            dq = _dls(J, twist6, lam=dls_lambda)

        # Enforce joint lock & step clamp
        if lock_joint_index is not None:
            dq[lock_joint_index] = 0.0
            q[lock_joint_index]  = cur[lock_joint_index]

        nrm = float(np.linalg.norm(dq))
        if nrm > max_step_rad:
            dq *= (max_step_rad / nrm)

        # Apply and (optionally) clamp to limits
        q = q + dq
        if joint_limits is not None:
            lo, hi = joint_limits
            q = np.minimum(np.maximum(q, np.asarray(lo, float)), np.asarray(hi, float))

        # Iterative refinement to meet plane tolerance
        it = 0
        while abs(err_n) > tol and it < max_corr_iters:
            pose = fk_func(q)
            p = np.array([pose.position.x, pose.position.y, pose.position.z], float)
            R_curr = _quat_to_R(pose.orientation)

            err_n = float(n_plane.dot(p - p_plane))
            pos_corr = -step_gain * err_n * n_plane

            J = np.asarray(jacobian_func(q), float)

            if J.shape[0] == 3:
                dq = _dls(J, pos_corr, lam=dls_lambda)
            else:
                R_err = R_des @ R_curr.T
                omega_err = SciR.from_matrix(R_err).as_rotvec()
                omega_plane = orient_gain * (axis_world0 * (omega_err @ axis_world0))
                twist6 = np.r_[pos_corr, omega_plane]

                if jacobian_frame.lower() == 'body':
                    Ad_inv = _adjoint_inv(R_curr, p)
                    twist6 = Ad_inv @ twist6
                if jacobian_order.lower() == 'ang_lin':
                    twist6 = np.r_[twist6[3:], twist6[:3]]

                dq = _dls(J, twist6, lam=dls_lambda)

            if lock_joint_index is not None:
                dq[lock_joint_index] = 0.0
                q[lock_joint_index]  = cur[lock_joint_index]

            nrm = float(np.linalg.norm(dq))
            if nrm > max_step_rad:
                dq *= (max_step_rad / nrm)

            q = q + dq
            if joint_limits is not None:
                lo, hi = joint_limits
                q = np.minimum(np.maximum(q, np.asarray(lo, float)), np.asarray(hi, float))

            it += 1

        corrected.append(q.tolist())

    print(f"🔢 Trajectory length after corrections: {len(corrected)}  (tol={tol} m, stride={corr_stride})")
    # Informative note when only 3xN J was available
    try:
        Jlast = np.asarray(jacobian_func(np.asarray(corrected[-1], float)))
        if theta_deg != 0.0 and Jlast.shape[0] == 3:
            print("⚠️ 3xN Jacobian: plane was enforced, but in-plane rotation (θ) wasn’t actively controlled.")
    except Exception:
        pass

    return corrected
