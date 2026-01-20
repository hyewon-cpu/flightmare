#!/usr/bin/env python3
"""
Interactive manual control script for Flightmare quadrotor.
Similar to Gazebo - you can see the visualization and control the drone with keyboard.

Usage:
    python interactive_control.py [--render]

Controls:
    W/S: Increase/Decrease forward thrust (pitch)
    A/D: Roll left/right
    Q/E: Yaw left/right
    Space: Increase altitude (all motors up)
    Shift: Decrease altitude (all motors down)
    R: Reset drone
    ESC/Q: Quit
"""

import os
import sys
import io
import argparse
import numpy as np
from ruamel.yaml import YAML
import time

# Try to import keyboard library for non-blocking input
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("Warning: 'keyboard' library not found. Install with: pip install keyboard")
    print("Falling back to blocking input (getch).")

from tonedio_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1


class InteractiveController:
    def __init__(self, render=True):
        self.render = render
        self.running = True
        
        # Action state (normalized [-1, 1] for each of 4 motors)
        # Actions represent motor thrusts: [motor0, motor1, motor2, motor3]
        # For quadrotor: front-left, front-right, back-right, back-left
        self.action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Control parameters
        self.base_thrust = 0.0  # Base hover thrust (will be adjusted)
        self.control_step = 0.05  # How much to change per keypress
        self.max_action = 1.0
        self.min_action = -1.0
        
        # Setup environment
        self.setup_env()
        
    def setup_env(self):
        """Initialize the Flightmare environment"""
        yaml = YAML()
        cfg_path = os.path.join(os.environ["FLIGHTMARE_PATH"], "flightlib/configs/vec_env.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.load(f)
        
        # Force single environment for interactive control
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
        cfg["env"]["render"] = "yes" if self.render else "no"
        
        # Convert to YAML string
        stream = io.StringIO()
        yaml.dump(cfg, stream)
        cfg_yaml_str = stream.getvalue()
        
        # Create environment
        self.env = wrapper.FlightEnvVec(QuadrotorEnv_v1(cfg_yaml_str, False))
        
        if self.render:
            print("Connecting to Unity...")
            self.env.connectUnity()
            print("Connected to Unity!")
        
        print(f"Environment initialized: obs_dim={self.env.num_obs}, act_dim={self.env.num_acts}")
        print("\n" + "="*60)
        print("INTERACTIVE CONTROL MODE")
        print("="*60)
        self.print_controls()
        
    def print_controls(self):
        """Print control instructions"""
        print("\nControls:")
        print("  W/S      : Pitch forward/backward")
        print("  A/D      : Roll left/right")
        print("  Q/E      : Yaw left/right")
        print("  Space    : Increase altitude (all motors up)")
        print("  Shift    : Decrease altitude (all motors down)")
        print("  R        : Reset drone")
        print("  ESC/Q    : Quit")
        print("\nNote: Actions are cumulative. Press keys multiple times for stronger control.")
        print("="*60 + "\n")
    
    def reset_action(self):
        """Reset action to neutral (hover)"""
        # Set to a small positive value to maintain altitude
        self.action = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        self.base_thrust = 0.1
    
    def update_action_from_keys(self):
        """Update action based on keyboard input"""
        if not HAS_KEYBOARD:
            return False
        
        action_changed = False
        
        # Pitch (forward/backward) - W/S
        if keyboard.is_pressed('w'):
            # Pitch forward: increase back motors, decrease front motors
            self.action[2] = min(self.max_action, self.action[2] + self.control_step)
            self.action[3] = min(self.max_action, self.action[3] + self.control_step)
            self.action[0] = max(self.min_action, self.action[0] - self.control_step)
            self.action[1] = max(self.min_action, self.action[1] - self.control_step)
            action_changed = True
        elif keyboard.is_pressed('s'):
            # Pitch backward: increase front motors, decrease back motors
            self.action[0] = min(self.max_action, self.action[0] + self.control_step)
            self.action[1] = min(self.max_action, self.action[1] + self.control_step)
            self.action[2] = max(self.min_action, self.action[2] - self.control_step)
            self.action[3] = max(self.min_action, self.action[3] - self.control_step)
            action_changed = True
        
        # Roll (left/right) - A/D
        if keyboard.is_pressed('a'):
            # Roll left: increase right motors, decrease left motors
            self.action[1] = min(self.max_action, self.action[1] + self.control_step)
            self.action[2] = min(self.max_action, self.action[2] + self.control_step)
            self.action[0] = max(self.min_action, self.action[0] - self.control_step)
            self.action[3] = max(self.min_action, self.action[3] - self.control_step)
            action_changed = True
        elif keyboard.is_pressed('d'):
            # Roll right: increase left motors, decrease right motors
            self.action[0] = min(self.max_action, self.action[0] + self.control_step)
            self.action[3] = min(self.max_action, self.action[3] + self.control_step)
            self.action[1] = max(self.min_action, self.action[1] - self.control_step)
            self.action[2] = max(self.min_action, self.action[2] - self.control_step)
            action_changed = True
        
        # Yaw (rotate) - Q/E
        if keyboard.is_pressed('q'):
            # Yaw left: increase counter-clockwise motors
            self.action[0] = min(self.max_action, self.action[0] + self.control_step)
            self.action[2] = min(self.max_action, self.action[2] + self.control_step)
            self.action[1] = max(self.min_action, self.action[1] - self.control_step)
            self.action[3] = max(self.min_action, self.action[3] - self.control_step)
            action_changed = True
        elif keyboard.is_pressed('e'):
            # Yaw right: increase clockwise motors
            self.action[1] = min(self.max_action, self.action[1] + self.control_step)
            self.action[3] = min(self.max_action, self.action[3] + self.control_step)
            self.action[0] = max(self.min_action, self.action[0] - self.control_step)
            self.action[2] = max(self.min_action, self.action[2] - self.control_step)
            action_changed = True
        
        # Altitude - Space/Shift
        if keyboard.is_pressed('space'):
            # Increase all motors
            self.action = np.clip(self.action + self.control_step, self.min_action, self.max_action)
            action_changed = True
        elif keyboard.is_pressed('shift'):
            # Decrease all motors
            self.action = np.clip(self.action - self.control_step, self.min_action, self.max_action)
            action_changed = True
        
        # Reset
        if keyboard.is_pressed('r'):
            self.reset_action()
            action_changed = True
            return 'reset'
        
        # Quit
        if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
            return 'quit'
        
        return action_changed
    
    def get_action_blocking(self):
        """Get action using blocking input (fallback if keyboard library not available)"""
        try:
            import sys
            import tty
            import termios
            
            def getch():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    ch = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return ch
            
            print("\nEnter command (w/a/s/d/q/e/space/shift/r/esc): ", end='', flush=True)
            key = getch()
            print(key)
            
            action_changed = False
            
            if key == 'w':
                self.action[2] = min(self.max_action, self.action[2] + self.control_step)
                self.action[3] = min(self.max_action, self.action[3] + self.control_step)
                self.action[0] = max(self.min_action, self.action[0] - self.control_step)
                self.action[1] = max(self.min_action, self.action[1] - self.control_step)
                action_changed = True
            elif key == 's':
                self.action[0] = min(self.max_action, self.action[0] + self.control_step)
                self.action[1] = min(self.max_action, self.action[1] + self.control_step)
                self.action[2] = max(self.min_action, self.action[2] - self.control_step)
                self.action[3] = max(self.min_action, self.action[3] - self.control_step)
                action_changed = True
            elif key == 'a':
                self.action[1] = min(self.max_action, self.action[1] + self.control_step)
                self.action[2] = min(self.max_action, self.action[2] + self.control_step)
                self.action[0] = max(self.min_action, self.action[0] - self.control_step)
                self.action[3] = max(self.min_action, self.action[3] - self.control_step)
                action_changed = True
            elif key == 'd':
                self.action[0] = min(self.max_action, self.action[0] + self.control_step)
                self.action[3] = min(self.max_action, self.action[3] + self.control_step)
                self.action[1] = max(self.min_action, self.action[1] - self.control_step)
                self.action[2] = max(self.min_action, self.action[2] - self.control_step)
                action_changed = True
            elif key == 'q':
                self.action[0] = min(self.max_action, self.action[0] + self.control_step)
                self.action[2] = min(self.max_action, self.action[2] + self.control_step)
                self.action[1] = max(self.min_action, self.action[1] - self.control_step)
                self.action[3] = max(self.min_action, self.action[3] - self.control_step)
                action_changed = True
            elif key == 'e':
                self.action[1] = min(self.max_action, self.action[1] + self.control_step)
                self.action[3] = min(self.max_action, self.action[3] + self.control_step)
                self.action[0] = max(self.min_action, self.action[0] - self.control_step)
                self.action[2] = max(self.min_action, self.action[2] - self.control_step)
                action_changed = True
            elif key == ' ':
                self.action = np.clip(self.action + self.control_step, self.min_action, self.max_action)
                action_changed = True
            elif key == '\x1b':  # ESC
                return 'quit'
            elif key == 'r':
                self.reset_action()
                return 'reset'
            
            return action_changed
            
        except Exception as e:
            print(f"Error in blocking input: {e}")
            return False
    
    def run(self):
        """Main control loop"""
        # Reset environment
        obs = self.env.reset()
        self.reset_action()
        
        step_count = 0
        last_print_time = time.time()
        
        print("Starting control loop...")
        print("Make sure Unity window is focused if using keyboard library.\n")
        
        try:
            while self.running:
                # Get keyboard input
                if HAS_KEYBOARD:
                    result = self.update_action_from_keys()
                    if result == 'quit':
                        break
                    elif result == 'reset':
                        obs = self.env.reset()
                        self.reset_action()
                        print("Drone reset!")
                        continue
                else:
                    # Use blocking input
                    result = self.get_action_blocking()
                    if result == 'quit':
                        break
                    elif result == 'reset':
                        obs = self.env.reset()
                        self.reset_action()
                        print("Drone reset!")
                        continue
                
                # Apply action (reshape to (1, 4) for single environment)
                action = self.action.reshape(1, -1)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Reset if done
                if done[0]:
                    print("Episode done, resetting...")
                    obs = self.env.reset()
                    self.reset_action()
                
                step_count += 1
                
                # Print status periodically
                current_time = time.time()
                if current_time - last_print_time > 1.0:  # Every 1 second
                    pos = obs[0, 0:3]
                    euler = obs[0, 3:6]
                    print(f"Step {step_count} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] | "
                          f"Euler: [{np.degrees(euler[0]):.1f}, {np.degrees(euler[1]):.1f}, {np.degrees(euler[2]):.1f}]° | "
                          f"Action: {self.action}")
                    last_print_time = current_time
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        if self.render:
            self.env.disconnectUnity()
        self.env.close()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Interactive manual control for Flightmare")
    parser.add_argument('--render', type=int, default=1,
                        help="Enable Unity rendering (1) or disable (0)")
    args = parser.parse_args()
    
    if not HAS_KEYBOARD:
        print("\nNOTE: For better experience, install keyboard library:")
        print("  pip install keyboard")
        print("  (May require sudo on Linux)")
        print("\nContinuing with blocking input mode...\n")
        time.sleep(2)
    
    controller = InteractiveController(render=bool(args.render))
    controller.run()


if __name__ == "__main__":
    main()
