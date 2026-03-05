

#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"


namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    goal_state_((Vector<quadenv::kNObs>() << 5.0, 7.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0)
                  .finished()) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  // Keep horizontal bounds, but allow higher-altitude hovering tasks.
  world_box_ << -30, 30, -30, 30, 0, 30;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // load parameters
  loadParam(cfg_);

  const char* reward_mode_env = std::getenv("FLIGHTMARE_REWARD_MODE");
  if (reward_mode_env != nullptr) {
    const std::string reward_mode(reward_mode_env);
    if (reward_mode == "hover") {
      forced_reward_mode_ = 0;
      logger_.info("Reward mode override: hover");
    } else if (reward_mode == "landing") {
      forced_reward_mode_ = 1;
      logger_.info("Reward mode override: landing");
    } else {
      forced_reward_mode_ = 0;
      logger_.warn("Unknown reward mode '%s', fallback to hover.", reward_mode.c_str());
    }
  }

  const char* action_mode_env = std::getenv("FLIGHTMARE_ACTION_MODE");
  if (action_mode_env != nullptr) {
    const std::string action_mode(action_mode_env);
    if (action_mode == "thrust_bodyrate") {
      use_ctbr_ = true;
      logger_.info("Action mode override: thrust_bodyrate");
    } else if (action_mode == "single_rotor_thrust") {
      use_ctbr_ = false;
      logger_.info("Action mode override: single_rotor_thrust");
    } else {
      logger_.warn("Unknown action mode '%s', keep YAML setting.", action_mode.c_str());
    }
  }
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();
  landing_phase_ = (forced_reward_mode_ == 1);

  if (random) {
    // Landing task: keep respawn close to a nominal start state.
    const Scalar pos_xy_noise = 0.5; //xy position noise 
    const Scalar pos_z_noise = 0.5;
    const Scalar vel_noise = 0.2;
    const Scalar att_noise = 0.05;
    const Scalar start_above_goal_z = 20.0;

    // reset position (near target XY, slightly above landing height)
    quad_state_.x(QS::POSX) =
      goal_state_(quadenv::kPos + 0) +
      pos_xy_noise * uniform_dist_(random_gen_) + spawn_offset_(0);
    quad_state_.x(QS::POSY) =
      goal_state_(quadenv::kPos + 1) +
      pos_xy_noise * uniform_dist_(random_gen_) + spawn_offset_(1);
    quad_state_.x(QS::POSZ) =
      goal_state_(quadenv::kPos + 2) + start_above_goal_z +
      pos_z_noise * uniform_dist_(random_gen_) + spawn_offset_(2);
    if (quad_state_.x(QS::POSZ) < 0.0) quad_state_.x(QS::POSZ) = 0.0;

    // reset linear velocity (small)
    quad_state_.x(QS::VELX) = vel_noise * uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = vel_noise * uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = vel_noise * uniform_dist_(random_gen_);

    // reset orientation (near level)
    quad_state_.x(QS::ATTW) = 1.0;
    quad_state_.x(QS::ATTX) = att_noise * uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = att_noise * uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = att_noise * uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  if (use_ctbr_) {
    cmd_.collective_thrust = collective_thrust_mean_;
    cmd_.omega.setZero();
    cmd_.thrusts = Vector<4>::Constant(NAN);
  } else {
    cmd_.collective_thrust = NAN;
    cmd_.omega = Vector<3>::Constant(NAN);
    cmd_.thrusts.setZero();
  }

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  if (use_ctbr_) {
    cmd_.collective_thrust = quad_act_(0);
    cmd_.omega = quad_act_.segment<3>(1);
    cmd_.thrusts = Vector<4>::Constant(NAN);
  } else {
    cmd_.collective_thrust = NAN;
    cmd_.omega = Vector<3>::Constant(NAN);
    cmd_.thrusts = quad_act_;
  }

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  // Hover phase uses a high-altitude goal, landing phase uses goal_state_.
  Vector<quadenv::kNObs> hover_goal_state = goal_state_;
  hover_goal_state(quadenv::kPos + 2) = 20.0;

  // ---------------------- hover reward (phase 1)
  Scalar pos_reward =
    pos_coeff_ * (quad_obs_.segment<quadenv::kNPos>(quadenv::kPos) -
                  hover_goal_state.segment<quadenv::kNPos>(quadenv::kPos))
                   .squaredNorm();
  Scalar ori_reward =
    ori_coeff_ * (quad_obs_.segment<quadenv::kNOri>(quadenv::kOri) -
                  hover_goal_state.segment<quadenv::kNOri>(quadenv::kOri))
                   .squaredNorm();
  Scalar lin_vel_reward =
    lin_vel_coeff_ * (quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel) -
                      hover_goal_state.segment<quadenv::kNLinVel>(quadenv::kLinVel))
                       .squaredNorm();
  Scalar ang_vel_reward =
    ang_vel_coeff_ * (quad_obs_.segment<quadenv::kNAngVel>(quadenv::kAngVel) -
                      hover_goal_state.segment<quadenv::kNAngVel>(quadenv::kAngVel))
                       .squaredNorm();
  Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();
  Scalar hover_total_reward =
    pos_reward + ori_reward + lin_vel_reward + ang_vel_reward + act_reward;
  hover_total_reward += 0.1;

  // ---------------------- landing reward (phase 2)
  const Vector<2> pos_xy = quad_state_.p.head<2>();
  const Vector<2> goal_xy = goal_state_.segment<2>(quadenv::kPos);
  const Scalar z = quad_state_.x(QS::POSZ);
  const Scalar goal_z = goal_state_(quadenv::kPos + 2);

  const Scalar xy_err_sq = (pos_xy - goal_xy).squaredNorm();
  const Scalar z_err = z - goal_z;
  const Scalar z_err_sq = z_err * z_err;
  const Scalar vel_sq = quad_state_.v.squaredNorm();
  const Scalar speed = std::sqrt(vel_sq);
  speed_log_sum_ += speed;
  speed_log_counter_++;
  if (speed_log_counter_ >= speed_log_interval_steps_) {
    const Scalar avg_speed =
      speed_log_sum_ / static_cast<Scalar>(speed_log_counter_);
    logger_.info("Average speed over last %d steps: %.4f m/s",
                 speed_log_counter_, avg_speed);
    speed_log_counter_ = 0;
    speed_log_sum_ = 0.0;
  }
  const Scalar roll = quad_obs_(quadenv::kOri + 2);
  const Scalar pitch = quad_obs_(quadenv::kOri + 1);
  const Scalar tilt_sq = roll * roll + pitch * pitch;
  const Scalar tilt = std::sqrt(tilt_sq);

  const Scalar w_xy = landing_w_xy_;
  const Scalar w_z = landing_w_z_;
  const Scalar w_vel = (z < landing_near_ground_z_) ? landing_w_vel_near_ : landing_w_vel_far_;
  const Scalar w_tilt = landing_w_tilt_;
  const Scalar tilt_soft = landing_tilt_soft_;
  const Scalar tilt_hard = landing_tilt_hard_;
  const Scalar w_tilt_excess = landing_w_tilt_excess_;
  const Scalar tilt_hard_penalty = landing_tilt_hard_penalty_;
  const Scalar time_penalty = landing_time_penalty_;
  const Scalar speed_soft_limit = landing_speed_soft_limit_;   // m/s
  const Scalar speed_hard_limit = landing_speed_hard_limit_;   // m/s
  const Scalar w_speed_excess = landing_w_speed_excess_;
  const Scalar hard_speed_penalty = landing_hard_speed_penalty_;

  Scalar shaping_reward = 0.0;
  shaping_reward -= w_xy * xy_err_sq;
  if (xy_err_sq < landing_xy_gate_ * landing_xy_gate_) {
    shaping_reward -= w_z * z_err_sq;
  } else {
    // If still far in XY, discourage descending too early.
    const Scalar descend_gap = goal_z + Scalar(2.0) - z;
    if (descend_gap > Scalar(0.0)) {
      shaping_reward -= landing_w_early_descend_ * descend_gap;
    }
  }
  shaping_reward -= w_vel * vel_sq;
  shaping_reward -= w_tilt * tilt_sq;
  if (tilt > tilt_soft) {
    const Scalar dt = tilt - tilt_soft;
    shaping_reward -= w_tilt_excess * dt * dt;
  }
  if (tilt > tilt_hard) {
    shaping_reward -= tilt_hard_penalty;
  }
  shaping_reward -= time_penalty;
  if (speed > speed_soft_limit) {
    const Scalar dv = speed - speed_soft_limit;
    shaping_reward -= w_speed_excess * dv * dv;
  }
  if (speed > speed_hard_limit) {
    shaping_reward -= hard_speed_penalty;
  }

  const Scalar landing_total_reward = shaping_reward + act_reward;
  return landing_phase_ ? landing_total_reward : hover_total_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  // Ground plane is assumed around z=3.0.
  if (quad_state_.x(QS::POSZ) <= landing_terminal_z_) {
    const Scalar xy_error =
      (quad_state_.p.head<2>() - goal_state_.segment<2>(quadenv::kPos)).norm();
    const Scalar vz = std::abs(quad_state_.x(QS::VELZ));
    const Vector<3> euler_zyx =
      quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
    const Scalar tilt = std::sqrt(
      euler_zyx(1) * euler_zyx(1) + euler_zyx(2) * euler_zyx(2));

    const bool success =
      (xy_error < landing_success_xy_error_) &&
      (vz < landing_success_vz_) &&
      (tilt < landing_success_tilt_);
    reward = success ? landing_success_reward_ : landing_failure_reward_;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_coeff_ = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }

  // Action mode configuration from YAML.
  use_ctbr_ = false;
  collective_thrust_mean_ = -Gz;
  collective_thrust_std_ = -Gz;
  bodyrate_mean_.setZero();
  bodyrate_std_.setConstant(6.0);
  if (cfg["quadrotor_env"] && cfg["quadrotor_env"]["action"]) {
    const YAML::Node acfg = cfg["quadrotor_env"]["action"];
    if (acfg["mode"]) {
      const std::string mode = acfg["mode"].as<std::string>();
      if (mode == "thrust_bodyrate") {
        use_ctbr_ = true;
      } else if (mode == "single_rotor_thrust") {
        use_ctbr_ = false;
      } else {
        logger_.warn("Unknown action mode '%s'. Using single_rotor_thrust.", mode.c_str());
      }
    }
    if (acfg["collective_thrust_mean"]) {
      collective_thrust_mean_ = acfg["collective_thrust_mean"].as<Scalar>();
    }
    if (acfg["collective_thrust_std"]) {
      collective_thrust_std_ = acfg["collective_thrust_std"].as<Scalar>();
    }
    if (acfg["bodyrate_mean"]) {
      const std::vector<Scalar> v = acfg["bodyrate_mean"].as<std::vector<Scalar>>();
      if (v.size() == 3) {
        bodyrate_mean_ = Vector<3>(v[0], v[1], v[2]);
      }
    }
    if (acfg["bodyrate_std"]) {
      const std::vector<Scalar> v = acfg["bodyrate_std"].as<std::vector<Scalar>>();
      if (v.size() == 3) {
        bodyrate_std_ = Vector<3>(v[0], v[1], v[2]);
      }
    }
  }

  // Action normalization depends on selected action mode.
  if (!use_ctbr_) {
    const Scalar mass = quadrotor_ptr_->getMass();
    act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
    act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;
  } else {
    act_mean_ << collective_thrust_mean_, bodyrate_mean_(0), bodyrate_mean_(1), bodyrate_mean_(2);
    act_std_ << collective_thrust_std_, bodyrate_std_(0), bodyrate_std_(1), bodyrate_std_(2);
  }
  logger_.info("Action mode: %s", use_ctbr_ ? "thrust_bodyrate" : "single_rotor_thrust");

  if (cfg["landing_reward"]) {
    const YAML::Node lcfg = cfg["landing_reward"];
    if (lcfg["w_xy"]) landing_w_xy_ = lcfg["w_xy"].as<Scalar>();
    if (lcfg["w_z"]) landing_w_z_ = lcfg["w_z"].as<Scalar>();
    if (lcfg["xy_gate"]) landing_xy_gate_ = lcfg["xy_gate"].as<Scalar>();
    if (lcfg["w_early_descend"]) landing_w_early_descend_ = lcfg["w_early_descend"].as<Scalar>();
    if (lcfg["w_vel_near"]) landing_w_vel_near_ = lcfg["w_vel_near"].as<Scalar>();
    if (lcfg["w_vel_far"]) landing_w_vel_far_ = lcfg["w_vel_far"].as<Scalar>();
    if (lcfg["near_ground_z"]) landing_near_ground_z_ = lcfg["near_ground_z"].as<Scalar>();
    if (lcfg["w_tilt"]) landing_w_tilt_ = lcfg["w_tilt"].as<Scalar>();
    if (lcfg["tilt_soft"]) landing_tilt_soft_ = lcfg["tilt_soft"].as<Scalar>();
    if (lcfg["tilt_hard"]) landing_tilt_hard_ = lcfg["tilt_hard"].as<Scalar>();
    if (lcfg["w_tilt_excess"]) landing_w_tilt_excess_ = lcfg["w_tilt_excess"].as<Scalar>();
    if (lcfg["tilt_hard_penalty"]) landing_tilt_hard_penalty_ = lcfg["tilt_hard_penalty"].as<Scalar>();
    if (lcfg["time_penalty"]) landing_time_penalty_ = lcfg["time_penalty"].as<Scalar>();
    if (lcfg["speed_soft_limit"]) landing_speed_soft_limit_ = lcfg["speed_soft_limit"].as<Scalar>();
    if (lcfg["speed_hard_limit"]) landing_speed_hard_limit_ = lcfg["speed_hard_limit"].as<Scalar>();
    if (lcfg["w_speed_excess"]) landing_w_speed_excess_ = lcfg["w_speed_excess"].as<Scalar>();
    if (lcfg["hard_speed_penalty"]) landing_hard_speed_penalty_ = lcfg["hard_speed_penalty"].as<Scalar>();
    if (lcfg["terminal_z"]) landing_terminal_z_ = lcfg["terminal_z"].as<Scalar>();
    if (lcfg["success_xy_error"]) landing_success_xy_error_ = lcfg["success_xy_error"].as<Scalar>();
    if (lcfg["success_vz"]) landing_success_vz_ = lcfg["success_vz"].as<Scalar>();
    if (lcfg["success_tilt"]) landing_success_tilt_ = lcfg["success_tilt"].as<Scalar>();
    if (lcfg["success_reward"]) landing_success_reward_ = lcfg["success_reward"].as<Scalar>();
    if (lcfg["failure_reward"]) landing_failure_reward_ = lcfg["failure_reward"].as<Scalar>();
  }
  return true;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
