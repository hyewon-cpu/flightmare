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
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    // Landing task: keep respawn close to a nominal start state.
    const Scalar pos_xy_noise = 0.5;
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
  cmd_.thrusts.setZero();

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
  cmd_.thrusts = quad_act_;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  // ---------------------- landing-centric shaping reward
  const Vector<2> pos_xy = quad_state_.p.head<2>();
  const Vector<2> goal_xy = goal_state_.segment<2>(quadenv::kPos);
  const Scalar z = quad_state_.x(QS::POSZ);
  const Scalar goal_z = goal_state_(quadenv::kPos + 2);

  const Scalar xy_err_sq = (pos_xy - goal_xy).squaredNorm();
  const Scalar z_err = z - goal_z;
  const Scalar vel_sq = quad_state_.v.squaredNorm();
  const Scalar speed = std::sqrt(vel_sq);
  const Scalar roll = quad_obs_(quadenv::kOri + 2);
  const Scalar pitch = quad_obs_(quadenv::kOri + 1);
  const Scalar tilt_sq = roll * roll + pitch * pitch;

  const Scalar w_xy = 1.0;
  const Scalar w_z = 0.3;
  const Scalar w_vel = (z < 1.0) ? 0.12 : 0.05;
  const Scalar w_tilt = 0.1;
  const Scalar time_penalty = 0.01;
  const Scalar speed_soft_limit = 1.5;   // m/s
  const Scalar speed_hard_limit = 3.0;   // m/s
  const Scalar w_speed_excess = 0.5;
  const Scalar hard_speed_penalty = 2.0;

  Scalar shaping_reward = 0.0;
  shaping_reward -= w_xy * xy_err_sq;
  shaping_reward -= w_z * z_err * z_err;
  shaping_reward -= w_vel * vel_sq;
  shaping_reward -= w_tilt * tilt_sq;
  shaping_reward -= time_penalty;
  if (speed > speed_soft_limit) {
    const Scalar dv = speed - speed_soft_limit;
    shaping_reward -= w_speed_excess * dv * dv;
  }
  if (speed > speed_hard_limit) {
    shaping_reward -= hard_speed_penalty;
  }

  // - control action penalty
  const Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();

  return shaping_reward + act_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  // Ground plane is assumed around z=3.0.
  if (quad_state_.x(QS::POSZ) <= 3.02) {
    const Scalar xy_error =
      (quad_state_.p.head<2>() - goal_state_.segment<2>(quadenv::kPos)).norm();
    const Scalar vz = std::abs(quad_state_.x(QS::VELZ));
    const Vector<3> euler_zyx =
      quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
    const Scalar tilt = std::sqrt(
      euler_zyx(1) * euler_zyx(1) + euler_zyx(2) * euler_zyx(2));

    const bool success =
      (xy_error < 0.30) && (vz < 0.50) && (tilt < 0.35);
    reward = success ? 10.0 : -5.0;
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
