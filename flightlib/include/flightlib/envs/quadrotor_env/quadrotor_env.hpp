#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

namespace quadenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kPos = 0,
  kNPos = 3,
  kOri = 3,
  kNOri = 3,
  kLinVel = 6,
  kNLinVel = 3,
  kAngVel = 9,
  kNAngVel = 3,
  kNObs = 12,
  // control actions
  kAct = 0,
  kNAct = 4,
};
};
class QuadrotorEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorEnv();
  QuadrotorEnv(const std::string &cfg_path);
  ~QuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);
  inline void setSpawnOffset(const Ref<Vector<3>> offset) {
    spawn_offset_ = offset;
  }

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnv"};

  // Define reward for training
  Scalar pos_coeff_, ori_coeff_, lin_vel_coeff_, ang_vel_coeff_, act_coeff_;

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;

  // reward function design (for model-free reinforcement learning)
  Vector<quadenv::kNObs> goal_state_;

  // action and observation normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
  Vector<3> spawn_offset_ = Vector<3>::Zero();

  // Reward mode selection.
  bool landing_phase_{false};
  // 0: hover reward, 1: landing reward
  int forced_reward_mode_{0};

  // Action mode:
  // false: single rotor thrusts [m0,m1,m2,m3]
  // true:  collective thrust + body rates [collective, wx, wy, wz]
  bool use_ctbr_{false};
  Scalar collective_thrust_mean_{0.0};
  Scalar collective_thrust_std_{0.0};
  Vector<3> bodyrate_mean_ = Vector<3>::Zero();
  Vector<3> bodyrate_std_ = Vector<3>::Ones();

  // Landing reward scalars (configurable via YAML: landing_reward)
  Scalar landing_w_xy_{0.05};
  Scalar landing_w_z_{0.03};
  Scalar landing_xy_gate_{1.0};
  Scalar landing_w_early_descend_{0.0};
  Scalar landing_w_vel_near_{0.05};
  Scalar landing_w_vel_far_{0.005};
  Scalar landing_near_ground_z_{4.0};
  Scalar landing_w_tilt_{0.05};
  Scalar landing_tilt_soft_{0.35};
  Scalar landing_tilt_hard_{0.70};
  Scalar landing_w_tilt_excess_{0.05};
  Scalar landing_tilt_hard_penalty_{0.5};
  Scalar landing_time_penalty_{0.002};
  Scalar landing_speed_soft_limit_{1.05};
  Scalar landing_speed_hard_limit_{3.0};
  Scalar landing_w_speed_excess_{0.05};
  Scalar landing_hard_speed_penalty_{0.02};
  Scalar landing_terminal_z_{3.02};
  Scalar landing_success_xy_error_{0.50};
  Scalar landing_success_vz_{0.80};
  Scalar landing_success_tilt_{0.45};
  Scalar landing_success_reward_{10.0};
  Scalar landing_failure_reward_{-5.0};

  // Debug logging: average speed per N steps.
  int speed_log_interval_steps_{10000};
  int speed_log_counter_{0};
  Scalar speed_log_sum_{0.0};
};

}  // namespace flightlib
