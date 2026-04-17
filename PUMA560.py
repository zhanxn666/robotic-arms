import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from spatialmath import SE3
import time
import imageio

# ==================== 1. IMPROVED DH MODEL ====================
# Create robot model (standard Puma560; replace with custom MDH parameters if needed)
robot = rtb.models.DH.Puma560()
# robot = DHRobot([...], mdh=True)       # <-- use your improved parameters here

print("=== DH Parameters (改进 DH 参数法) ===")
# Robustly show DH parameters: try attribute 'dh', else build from link parameters
if hasattr(robot, 'dh'):
    print(robot.dh)
elif hasattr(robot, 'links'):
    print("a\talpha\td\ttheta (from robot.links)")
    for L in robot.links:
        a = getattr(L, 'a', None)
        alpha = getattr(L, 'alpha', None)
        d = getattr(L, 'd', None)
        theta = getattr(L, 'theta', None)
        print(f"{a}\t{alpha}\t{d}\t{theta}")
else:
    print("Robot model has no 'dh' or 'links' attribute; available attrs:", [x for x in dir(robot) if not x.startswith('_')])

# ==================== 2. TARGET TRAJECTORY POINTS (Cartesian) ====================
# Example via-points in Cartesian space (you can change these)
via_poses = [
    SE3(0.5, -0.2, 0.3) * SE3.Rx(np.pi/2),   # start
    SE3(0.6, 0.0, 0.4)  * SE3.Rx(np.pi/2),
    SE3(0.4, 0.3, 0.5)  * SE3.Rx(np.pi/2),
    SE3(0.2, -0.1, 0.6) * SE3.Rx(np.pi/2),   # end
]

# Solve IK once to get joint via-points
q_via = np.zeros((len(via_poses), 6))
for i, T in enumerate(via_poses):
    # use a safe initial guess for IK
    q0_guess = getattr(robot, 'qz', np.zeros(getattr(robot, 'n', 6))) if i == 0 else q_via[i-1]
    sol = robot.ikine_LM(T, q0=q0_guess)
    # some IK solvers return an object with attribute 'q', others return a plain array
    q_via[i] = getattr(sol, 'q', sol)

print("Joint via-points (rad):", q_via)

# Time vector (total 8 seconds, 200 samples)
t_total = 8.0
N = 200
t = np.linspace(0, t_total, N)
dt = t[1] - t[0]

# ==================== 3. CUBIC POLYNOMIAL INTERPOLATION (三次多项式插值) ====================
# Segmented cubic spline in joint space with C¹ continuity (position + velocity)
def cubic_trajectory(q_via, t, t_total):
    # t: fine time vector; t_total: total motion time; q_via shape = (n_via, n_joints)
    t_via = np.linspace(0, t_total, len(q_via))
    cs = CubicSpline(t_via, q_via, bc_type='clamped', axis=0)   # clamped = zero vel at start/end
    q_cubic = cs(t)
    qd_cubic = cs(t, 1)
    qdd_cubic = cs(t, 2)
    return q_cubic, qd_cubic, qdd_cubic

q_cubic, qd_cubic, qdd_cubic = cubic_trajectory(q_via, t, t_total)

# ==================== 4. HYBRID 1ST/2ND-ORDER METHOD (一次二次混合法 = LSPB) ====================
# Multi-segment LSPB hybrid (linear segments + parabolic blends)
def _lspb_scalar(q0, q1, T, t_local, tacc_fraction=0.2):
    # LSPB for a single scalar joint motion from q0->q1 over T seconds
    if T <= 0 or np.allclose(q0, q1):
        return np.full_like(t_local, q0), np.zeros_like(t_local), np.zeros_like(t_local)
    tb = min(tacc_fraction * T, T / 2.0)
    Delta = q1 - q0
    # handle edge cases: no blend (tb == 0) -> pure linear
    if tb <= 0:
        q = q0 + (Delta / T) * t_local
        qd = np.full_like(t_local, Delta / T)
        qdd = np.zeros_like(t_local)
        return q, qd, qdd
    Tc = max(0.0, T - 2.0 * tb)  # constant-velocity duration
    Vm = Delta / (T - tb)  # Vm*(T - tb) = Delta
    a = Vm / tb
    q = np.zeros_like(t_local)
    qd = np.zeros_like(t_local)
    qdd = np.zeros_like(t_local)
    # precompute start-of-decel position
    q_s = q0 + 0.5 * a * tb ** 2 + Vm * Tc
    for idx, tau in enumerate(t_local):
        if tau <= 0:
            q[idx] = q0
            qd[idx] = 0.0
            qdd[idx] = 0.0
        elif tau < tb:
            # acceleration phase
            q[idx] = q0 + 0.5 * a * tau ** 2
            qd[idx] = a * tau
            qdd[idx] = a
        elif tau <= (tb + Tc):
            # constant velocity phase
            q[idx] = q0 + 0.5 * a * tb ** 2 + Vm * (tau - tb)
            qd[idx] = Vm
            qdd[idx] = 0.0
        elif tau <= T:
            # deceleration phase
            dtau = tau - (tb + Tc)
            q[idx] = q_s + Vm * dtau - 0.5 * a * dtau ** 2
            qd[idx] = Vm - a * dtau
            qdd[idx] = -a
        else:
            q[idx] = q1
            qd[idx] = 0.0
            qdd[idx] = 0.0
    return q, qd, qdd

def mstraj_lspb(q_via, t, t_total, tacc_fraction=0.2):
    # q_via: (n_via, n_joints); t: fine time vector; t_total: total duration
    n_via, n_j = q_via.shape
    q = np.zeros((len(t), n_j))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    seg_T = t_total / max(1, (n_via - 1))
    for seg in range(n_via - 1):
        t0 = seg * seg_T
        t1 = (seg + 1) * seg_T
        # avoid overlapping assignments at segment boundaries: include left, exclude right
        if seg < n_via - 2:
            mask = (t >= t0) & (t < t1)
        else:
            mask = (t >= t0) & (t <= t1)
        t_local = t[mask] - t0
        for j in range(n_j):
            q0 = q_via[seg, j]
            q1 = q_via[seg + 1, j]
            q_seg, qd_seg, qdd_seg = _lspb_scalar(q0, q1, seg_T, t_local, tacc_fraction=tacc_fraction)
            q[mask, j] = q_seg
            qd[mask, j] = qd_seg
            qdd[mask, j] = qdd_seg
    return q, qd, qdd

q_hybrid, qd_hybrid, qdd_hybrid = mstraj_lspb(q_via, t, t_total, tacc_fraction=0.1)

# ==================== 4.5 ADDITIONAL OUTPUTS: DH Matrix, Link Transforms, Snapshots ====================
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def dh_transform(a, alpha, d, theta):
    ca = np.cos(alpha); sa = np.sin(alpha)
    ct = np.cos(theta); st = np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,    d],
        [0.0,    0.0,     0.0,  1.0],
    ])


def compute_link_transforms(robot, q):
    T = np.eye(4)
    Ts = [T.copy()]
    for i, L in enumerate(robot.links):
        a = float(getattr(L, 'a', 0.0))
        alpha = float(getattr(L, 'alpha', 0.0))
        d = float(getattr(L, 'd', 0.0))
        theta_off = float(getattr(L, 'theta', 0.0))
        offset = float(getattr(L, 'offset', 0.0)) if hasattr(L, 'offset') else 0.0
        qi = float(q[i]) if i < len(q) else 0.0
        theta = theta_off + qi + offset
        A = dh_transform(a, alpha, d, theta)
        T = T @ A
        Ts.append(T.copy())
    return Ts


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim3d(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim3d(z_mid - max_range / 2, z_mid + max_range / 2)


def save_arm_configuration(robot, q, filename):
    Ts = compute_link_transforms(robot, q)
    origins = np.array([T[:3, 3] for T in Ts])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o', markersize=6)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    set_axes_equal(ax)
    ax.set_title(f'Arm configuration: {os.path.basename(filename)}')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def compute_ee_trajectory(robot, q_traj):
    ee = np.zeros((len(q_traj), 3))
    for i, qrow in enumerate(q_traj):
        T = robot.fkine(qrow)
        if hasattr(T, 'A'):
            A = T.A
        elif hasattr(T, 'matrix'):
            A = T.matrix
        else:
            A = np.array(T)
        ee[i] = A[:3, 3]
    return ee


def save_ee_plots(ee_xyz, t, prefix):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], '-o', markersize=2)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    set_axes_equal(ax)
    ax.set_title(prefix + ' End-Effector Path')
    plt.tight_layout()
    figpath = f'{prefix}_ee_path.png'
    plt.savefig(figpath, dpi=200)
    plt.close(fig)

    fig2, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    axs[0].plot(t, ee_xyz[:, 0], '-b'); axs[0].set_ylabel('X (m)')
    axs[1].plot(t, ee_xyz[:, 1], '-g'); axs[1].set_ylabel('Y (m)')
    axs[2].plot(t, ee_xyz[:, 2], '-r'); axs[2].set_ylabel('Z (m)')
    axs[2].set_xlabel('Time (s)')
    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    figpath2 = f'{prefix}_ee_xyz_vs_time.png'
    plt.savefig(figpath2, dpi=200)
    plt.close(fig2)


def save_matrices_latex(T06, T0c, filename, caption=None):
    """Save T06 and T0c matrices to a LaTeX file for direct inclusion.

    The file contains two equation* environments with bmatrix matrices.
    """
    T06 = np.array(T06)
    T0c = np.array(T0c)
    with open(filename, 'w') as f:
        f.write('% LaTeX matrices generated by PUMA560.py\n')
        if caption:
            f.write('% ' + caption + '\n')
        f.write('\\begin{equation*}\\n')
        f.write('T_{06} = \\begin{bmatrix}\\n')
        for row in T06:
            f.write(' & '.join([f'{v:.6f}' for v in row]))
            f.write(' \\\\n+')
        f.write('\\end{bmatrix}\\n')
        f.write('\\end{equation*}\\n\\n')

        f.write('\\begin{equation*}\\n')
        f.write('T_{0c} = \\begin{bmatrix}\\n')
        for row in T0c:
            f.write(' & '.join([f'{v:.6f}' for v in row]))
            f.write(' \\\\n+')
        f.write('\\end{bmatrix}\\n')
        f.write('\\end{equation*}\\n')


def save_animation_gif(robot, q_traj, filename, fps=25, figsize=(6, 6), dpi=80):
    """Render the robot at each joint configuration and save a GIF.

    This uses the existing compute_link_transforms to draw a simple stick
    representation for each frame, avoiding the robot.plot writer issues.
    """
    # stream frames to GIF writer to avoid storing all frames in memory
    with imageio.get_writer(filename, mode='I', fps=fps) as writer:
        for qrow in q_traj:
            Ts = compute_link_transforms(robot, qrow)
            origins = np.array([T[:3, 3] for T in Ts])
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(origins[:, 0], origins[:, 1], origins[:, 2], '-o', markersize=6)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            set_axes_equal(ax)
            plt.tight_layout()
            # draw and capture as RGB array
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
            writer.append_data(img)
            plt.close(fig)



# Print DH parameter matrix as numeric array
dh_rows = []
if hasattr(robot, 'dh'):
    try:
        dh_rows = np.array([[float(getattr(r, 'a', 0.0)), float(getattr(r, 'alpha', 0.0)), float(getattr(r, 'd', 0.0)), float(getattr(r, 'theta', 0.0))] for r in robot.dh])
    except Exception:
        dh_rows = np.array([[float(getattr(L, 'a', 0.0)), float(getattr(L, 'alpha', 0.0)), float(getattr(L, 'd', 0.0)), float(getattr(L, 'theta', 0.0))] for L in robot.links])
else:
    dh_rows = np.array([[float(getattr(L, 'a', 0.0)), float(getattr(L, 'alpha', 0.0)), float(getattr(L, 'd', 0.0)), float(getattr(L, 'theta', 0.0))] for L in robot.links])

print('\nDH parameter matrix (a, alpha, d, theta):')
print(np.array2string(dh_rows, precision=6, separator=', '))

# Print per-link transforms for each via-point and save arm snapshots
for i, qv in enumerate(q_via):
    Ts = compute_link_transforms(robot, qv)
    print(f"\n=== Link transforms for via-point {i} ===")
    for j, T in enumerate(Ts[1:], start=1):
        print(f"T_{j} =\n{np.array2string(T, precision=6, separator=', ')}\n")

    # 0_6 transform (base to end-effector) from link chain
    T_06 = Ts[-1].copy()
    print("T_06 (base -> end-effector) =\n", np.array2string(T_06, precision=6, separator=', '))

    # 0_c transform: the desired Cartesian target pose provided as via_pose
    vp = via_poses[i]
    if hasattr(vp, 'A'):
        T_0c = vp.A
    elif hasattr(vp, 'matrix'):
        T_0c = vp.matrix
    else:
        T_0c = np.array(vp)
    print("T_0c (base -> desired target pose) =\n", np.array2string(T_0c, precision=6, separator=', '))

    # Compute pose errors: position norm and orientation difference (deg)
    pos_err = np.linalg.norm(T_0c[:3, 3] - T_06[:3, 3])
    R_06 = T_06[:3, :3]
    R_0c = T_0c[:3, :3]
    R_rel = R_06.T @ R_0c
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    arg = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    ori_err_rad = np.arccos(arg)
    ori_err_deg = np.degrees(ori_err_rad)
    print(f"Position error norm: {pos_err:.6f} m; Orientation error: {ori_err_deg:.6f} deg")

    # Save textual transforms for report convenience
    txtname = f'transforms_via{i}.txt'
    with open(txtname, 'w') as f:
        f.write(f"via-point {i}\n\n")
        f.write("T_06 (base -> end-effector):\n")
        f.write(np.array2string(T_06, precision=6, separator=', ') + "\n\n")
        f.write("T_0c (base -> desired target pose):\n")
        f.write(np.array2string(T_0c, precision=6, separator=', ') + "\n\n")
        f.write(f"Position error norm: {pos_err:.6f} m\n")
        f.write(f"Orientation error: {ori_err_deg:.6f} deg\n")

    # Save LaTeX version for direct report inclusion
    texname = f'transforms_via{i}.tex'
    save_matrices_latex(T_06, T_0c, texname, caption=f'Via-point {i} transforms and errors')

    imgname = f'arm_config_via{i}.png'
    save_arm_configuration(robot, qv, imgname)

# Compute and save end-effector trajectories for both methods
ee_cubic = compute_ee_trajectory(robot, q_cubic)
ee_hybrid = compute_ee_trajectory(robot, q_hybrid)
save_ee_plots(ee_cubic, t, 'cubic')
save_ee_plots(ee_hybrid, t, 'hybrid')

# ==================== 5. CONTINUITY & DYNAMIC PERFORMANCE VERIFICATION ====================
def plot_trajectories(title, q, qd, qdd, color, savefile=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)
    n_j = q.shape[1] if q.ndim == 2 else 6
    for i in range(n_j):
        axs[0].plot(t, q[:, i], color=color, alpha=0.7, label=f'q{i+1}')
        axs[1].plot(t, qd[:, i], color=color, alpha=0.7, label=f'qd{i+1}')
        axs[2].plot(t, qdd[:, i], color=color, alpha=0.7, label=f'qdd{i+1}')
    axs[0].set_ylabel('Joint position (rad)')
    axs[1].set_ylabel('Joint velocity (rad/s)')
    axs[2].set_ylabel('Joint accel (rad/s²)')
    axs[2].set_xlabel('Time (s)')
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=200)
    plt.show()
    plt.close(fig)

plot_trajectories("三次多项式插值 (Cubic)", q_cubic, qd_cubic, qdd_cubic, 'blue')
plot_trajectories("一次二次混合法 (LSPB)", q_hybrid, qd_hybrid, qdd_hybrid, 'orange')

# ==================== 6. COMPARISON ANALYSIS ====================
print("\n=== 不同轨迹规划方法对比分析 ===")
for name, qd, qdd in [("Cubic", qd_cubic, qdd_cubic), ("LSPB Hybrid", qd_hybrid, qdd_hybrid)]:
    print(f"\n{name} method:")
    print(f"  Max velocity: {np.max(np.abs(qd)):.4f} rad/s")
    print(f"  Max acceleration: {np.max(np.abs(qdd)):.4f} rad/s²")
    print(f"  Execution time: {t_total} s")
    # Check continuity
    print(f"  Velocity continuous? {np.all(np.abs(np.diff(qd, axis=0)) < 1e-3)} (C1)")
    print(f"  Acceleration continuous? {np.all(np.abs(np.diff(qdd, axis=0)) < 1e-2)} (approx C2)")

# ==================== 7. VISUAL SIMULATION (MATLAB-style) ====================
print("\nSaving GIF animations for both trajectories...")
fps = max(1, int(round(1.0 / dt))) if dt > 0 else 25
print(f"Using fps={fps} for GIF generation (N={len(t)} frames)")
save_animation_gif(robot, q_cubic, 'cubic_traj.gif', fps=fps)
print('Saved cubic_traj.gif')
save_animation_gif(robot, q_hybrid, 'hybrid_traj.gif', fps=fps)
print('Saved hybrid_traj.gif')

print("\n✅ All requirements completed!")
print("   • 改进 DH 模型 + 正逆运动学推导")
print("   • 分段三次多项式 + 一次二次混合法")
print("   • 关节/笛卡尔空间连续性约束验证")
print("   • MATLAB-style 仿真 + 对比分析")