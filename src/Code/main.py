import os
import argparse
import numpy as np
import cv2
from scipy.integrate import solve_ivp
from pyquaternion import Quaternion

from splat_render import SplatRenderer
from control import QuadrotorController
from quad_dynamics import model_derivative
import tello
import raft_segment
from seg import find_largest_gap_from_flow


# ============================================================
# Quadrotor Navigation Function
# ============================================================
def goToWaypoint(currentPose, targetPose, velocity=0.01):
    """
    Navigate quadrotor to a target waypoint

    Parameters:
    - currentPose: {'position': [x,y,z], 'rpy':[r,p,y]} in NED (meters, radians)
    - targetPose: [x, y, z] target position in NED frame (meters)
    - velocity: cruise velocity (m/s)

    Returns:
    - newPose: {'position': final_pos, 'rpy': final_rpy}
    """
    dt = 0.01   # 10ms timestep
    tolerance = 0.03  # 3 cm tolerance
    max_time = 30.0   # Maximum 30 seconds

    # Initialize controller
    controller = QuadrotorController(tello)
    param = tello

    # Extract current state
    pos = np.array(currentPose['position'])
    rpy = np.array(currentPose['rpy'])  # roll, pitch, yaw in radians

    # Initialize velocities to zero (starting from hover)
    vel = np.zeros(3)
    pqr = np.zeros(3)

    # Convert roll, pitch, yaw to quaternion (body w.r.t NED)
    roll, pitch, yaw = rpy
    quat = (Quaternion(axis=[0, 0, 1], radians=yaw) *
            Quaternion(axis=[0, 1, 0], radians=pitch) *
            Quaternion(axis=[1, 0, 0], radians=roll))

    # Build state vector [x, y, z, vx, vy, vz, qx, qy, qz, qw, p, q, r]
    current_state = np.concatenate(
        [pos, vel, [quat.x, quat.y, quat.z, quat.w], pqr]
    )

    target_position = np.array(targetPose)

    # Calculate distance and estimated time
    distance = np.linalg.norm(target_position - pos)
    estimated_time = min(max(distance, 1e-6) / max(velocity, 1e-6) * 2.0, max_time)

    print(f"  Navigating: {pos} → {target_position}")
    print(f"  Distance: {distance:.2f}m, Est. time: {estimated_time:.1f}s")

    # Check if already at target
    if distance < tolerance:
        print("  Already at target!")
        return {'position': pos, 'rpy': rpy}

    # Generate trajectory
    num_points = max(2, int(estimated_time / dt))
    time_points = np.linspace(0, estimated_time, num_points)

    # Create trajectory with trapezoidal velocity profile
    direction = target_position - pos
    unit_direction = direction / max(distance, 1e-6)

    trajectory_points = []
    velocities = []
    accelerations = []

    accel_time = min(1.0, estimated_time * 0.25)
    decel_time = accel_time
    cruise_time = max(0.0, estimated_time - accel_time - decel_time)

    cruise_vel = min(
        velocity,
        distance / (0.5 * accel_time + cruise_time + 0.5 * decel_time + 1e-6)
    )

    for t in time_points:
        if t <= accel_time:
            # Acceleration phase
            vel_mag = (cruise_vel / accel_time) * t
            acc_mag = cruise_vel / accel_time
            progress = 0.5 * (cruise_vel / accel_time) * t * t / max(distance, 1e-6)
        elif t <= accel_time + cruise_time:
            # Cruise phase
            vel_mag = cruise_vel
            acc_mag = 0.0
            progress = (0.5 * cruise_vel * accel_time +
                        cruise_vel * (t - accel_time)) / max(distance, 1e-6)
        else:
            # Deceleration phase
            t_decel = t - accel_time - cruise_time
            vel_mag = cruise_vel - (cruise_vel / decel_time) * t_decel
            acc_mag = -cruise_vel / decel_time
            progress = (0.5 * cruise_vel * accel_time +
                        cruise_vel * cruise_time +
                        cruise_vel * t_decel -
                        0.5 * (cruise_vel / decel_time) * t_decel * t_decel) / max(distance, 1e-6)

        progress = np.clip(progress, 0.0, 1.0)
        position = pos + progress * direction
        vel_vec = vel_mag * unit_direction
        acc_vec = acc_mag * unit_direction

        trajectory_points.append(position)
        velocities.append(vel_vec)
        accelerations.append(acc_vec)

    trajectory_points = np.array(trajectory_points)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    # Set trajectory in controller
    controller.set_trajectory(trajectory_points, time_points,
                              velocities, accelerations)

    # Simulation loop
    state = current_state.copy()
    state_final = state

    for i, t in enumerate(time_points):
        # Compute control input
        control_input = controller.compute_control(state, t)

        # Check if reached
        current_pos = state[0:3]
        error = np.linalg.norm(current_pos - target_position)
        if error < tolerance and t > 1.0:
            print(f"  ✓ Reached at t={t:.2f}s, error={error:.3f}m")
            state_final = state
            break

        # Integrate dynamics
        if i < len(time_points) - 1:
            sol = solve_ivp(
                lambda t_, X: model_derivative(t_, X, control_input, param),
                [t, t + dt],
                state,
                method='RK45',
                max_step=dt
            )
            state = sol.y[:, -1]
            state_final = state
    else:
        # Loop completed without break
        error = np.linalg.norm(state_final[0:3] - target_position)
        print(f"  Final error: {error:.3f}m")

    # Extract final pose
    final_pos = state_final[0:3]
    final_quat = Quaternion(state_final[9],  # w
                            state_final[6],  # x
                            state_final[7],  # y
                            state_final[8])  # z
    final_ypr = final_quat.yaw_pitch_roll  # [yaw, pitch, roll]
    final_rpy = np.array([final_ypr[2], final_ypr[1], final_ypr[0]])  # [roll, pitch, yaw]

    newPose = {
        'position': final_pos,
        'rpy': final_rpy
    }
    return newPose


# ============================================================
# Helper: Compute step from gap center
# ============================================================
def compute_alignment_step(center,
                           img_shape,
                           forward_step=0.05,
                           max_lateral_step=0.06,
                           max_vertical_step=0.06,
                           gain_y=0.30,
                           gain_z=0.30):
    """
    Map gap center in image to NED increments.

    NED convention:
      x: forward
      y: right
      z: down

    Image convention (OpenCV):
      u: x-axis, increases to the right
      v: y-axis, increases downward
delta_ned
    We NEGATE the horizontal gain so that:
      - if the gap is to the RIGHT in the image (u > center),
        we move LEFT in NED (y negative) to re-center.
    """
    if center is None:
        # No alignment info: only move forward
        return np.array([forward_step, 0.0, 0.0])

    H, W = img_shape[:2]
    img_cx = W / 2.0  # Image center x (horizontal)
    img_cy = H / 2.0  # Image center y (vertical)

    u, v = center
    
    # Compute normalized offsets in [-1, 1]
    offset_u = (u - img_cx) / (W / 2.0)  # Horizontal offset
    offset_v = (v - img_cy) / (H / 2.0)  # Vertical offset
    

    # Forward step is fixed
    dx = forward_step

    dy = -gain_y * offset_u  # REMOVED the negation!

    dz = gain_z * offset_v

    # Clamp corrections to safety limits
    dy = np.clip(dy, -max_lateral_step, max_lateral_step)
    dz = np.clip(dz, -max_vertical_step, max_vertical_step)

    return np.array([dx, dy, dz])

# ============================================================
# Main: RAFT Warm-up Pattern + Visual Servoing
# ============================================================
def main(renderer,
         raft_model_path="./RAFT/models/raft-things.pth",
         output_dir="./Images",
         flow_dir="./Flow"):

    print("[MAIN] Starting main()")

    # Setup
    os.makedirs('./log', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
   

    # --------------------------------------------------------
    # INITIAL POSE (NED)
    # --------------------------------------------------------
    currentPose = {
        'position': np.array([0.0, 0.22, 0.0]),            # [x,y,z] in meters (NED)
        'rpy': np.radians([0.0, 0.0, 0.0])                # [roll,pitch,yaw]
    }

    # --------------------------------------------------------
    # FIRST FRAME: step_00 (for RAFT warm-up pattern)
    # --------------------------------------------------------
    color_image, depth_image, metric_depth = renderer.render(
        currentPose['position'],
        currentPose['rpy']
    )
    step_0_path = os.path.join(output_dir, "step_00.png")
    cv2.imwrite(step_0_path, color_image)
    cv2.imwrite('./log/initial_color.png', color_image)
    print("[MAIN] Saved step_00.png")

    # Your original motion patteimg_prev_path = step_0_path= 5
    max_pattern_iters = 16
    forward_step = 0.04    
    max_lateral_step=0.06
    max_vertical_step=0.06# meters per servo step
    center_jump_thresh_px = 80.0  # max allowed jump between frames
    img_prev_path = step_0_path
    prev_center = None
    currentPose = {
        'position': np.array([0.0, 0.0, 0.0]),            # [x,y,z] in meters (NED)
        'rpy': np.radians([0.0, 0.0, 0.0])                # [roll,pitch,yaw]
    }
    # img_prev_path is already last pattern frame
    for step_idx in range(1,max_pattern_iters):
        print(f"\n========== SERVO STEP {step_idx} (frame {step_idx:02d}) ==========")

        if (step_idx > 15):
            forward_step = 0.01
            max_lateral_step=0.12
            max_vertical_step=0.12
        
        # Level the drone for render
        currentPose['rpy'][0] = 0.0
        currentPose['rpy'][1] = 0.0
        currentPose['rpy'][2] = 0.0
        currentPose['position'][2] = -0.02
               

        # Render current view
        color_image, depth_image, metric_depth = renderer.render(
            currentPose['position'],
            currentPose['rpy']
        )

        # Save current RGB frame
        img_curr_path = os.path.join(output_dir, f"step_{step_idx:02d}.png")
        cv2.imwrite(img_curr_path, color_image)
        print(f"[SERVO] Saved {img_curr_path}")

        # Run RAFT on (prev, curr)
        raft_segment.run_raft_on_pair(
            img_prev_path,
            img_curr_path,
            raft_model_path,
            step_idx
        )
        
        flow_vis_path = f"./output/{step_idx:02d}.png"
        # Load flow visualization
        flow_img = cv2.imread(flow_vis_path, cv2.IMREAD_COLOR)
        
        if flow_img is None:
            print(f"[WARN] Could not load flow image: {flow_vis_path}")
            current_center = None
            mask_main = None
        else:
            # seg.find_largest_gap_from_flow(flow_img) → (mask_main, center)
            mask_main, current_center = find_largest_gap_from_flow(flow_img)

            # Save mask for debugging
            if mask_main is not None:
                mask_path = os.path.join(flow_dir, f"mask_{step_idx:02d}.png")
                cv2.imwrite(mask_path, mask_main)
                print(f"[SERVO] Saved mask {mask_path}")

        # Temporal consistency on gap center
        used_center = None
        if current_center is not None:
            if prev_center is None:
                used_center = current_center
            else:
                dist = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                if dist <= center_jump_thresh_px:
                    used_center = current_center
                else:
                    print(f"[WARN] Gap center jump too large ({dist:.1f}px). "
                          f"Reusing previous center.")
                    used_center = prev_center
        else:
            if prev_center is not None:
                print("[WARN] No gap detected. Using previous center.")
                used_center = prev_center
            else:
                print("[WARN] No gap detected yet and no previous center. "
                      "Moving purely forward.")
                used_center = None

        prev_center = used_center

        if used_center is not None:
            print(f"  Using gap center: ({used_center[0]:.1f}, {used_center[1]:.1f})")
        else:
            print("  No valid gap center, forward-only motion.")

        # Compute NED step from center + forward
        if flow_img is not None:
            img_shape = flow_img.shape
        else:
            img_shape = color_image.shape

        delta_ned = compute_alignment_step(
            used_center,
            img_shape,
            forward_step=forward_step
        )
        print(f"  Step delta (NED): {delta_ned}")

        target_position = currentPose['position'] + delta_ned
        # print(f"  Target position: {target_position}")
        target_position[2] = -0.01
        print(f"  Target position: {target_position}")
        
        # Move quadrotor
        currentPose = goToWaypoint(currentPose, target_position, velocity=0.02)

        # Update prev frame for next iteration
        img_prev_path = img_curr_path
        
        if(step_idx == (max_pattern_iters-1)):
            for i in range(6):
                currentPose['position'][0] += 0.05
                currentPose['rpy'][0] = 0.0
                currentPose['rpy'][1] = 0.0
                currentPose['rpy'][2] = 0.0
                currentPose['position'][1] += -0.003
                currentPose['position'][2] = -0.06
                
                color_image_final, depth_image_final, metric_depth_final = renderer.render(
                    currentPose['position'],
                    currentPose['rpy']
                )
                cv2.imwrite(f"./Images/step_{step_idx+1+i:02d}.png", color_image_final)


    print(f'\nSaved final frame at position {currentPose["position"]}')
    print("[MAIN] Done.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="../data/p4_colmap_nov6_1000_splat/p4_colmap_nov6_1000/splatfacto/2025-11-06_161816/config.yml",
        type=str
    )
    parser.add_argument(
        "--settings",
        default="../render_settings/render_settings.json",
        type=str
    )
    parser.add_argument(
        "--raft_model",
        default="./RAFT/models/raft-things.pth",
        type=str
    )
    args = parser.parse_args()

    print("[MAIN] Creating renderer...")
    renderer = SplatRenderer(args.config, args.settings)
    print("[MAIN] Renderer created. Entering main()...")
    main(renderer, raft_model_path=args.raft_model)

