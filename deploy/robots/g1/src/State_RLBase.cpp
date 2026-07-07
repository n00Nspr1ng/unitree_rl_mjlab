#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cstdio>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

// Scripted whole-body command (13-dim: [vx, vy, wz, l_ee(3), r_ee(3), rpy(3), height]),
// mirroring scripts/sim2sim.py PRESET_SEQUENCES. Cycles a preset sequence by segment over time
// and low-pass smooths (first-order, tau seconds) so segment boundaries ramp instead of step --
// abrupt command jumps are unsafe on hardware. Replaces the separate velocity_commands(3) +
// wb_command(10) obs, which are adjacent in the obs so a single 13-dim term is byte-identical.
// params: {sequence: <name>, segment_s: <float>, smoothing_tau: <float>}.
REGISTER_OBSERVATION(command)
{
    using Row = std::array<float, 13>;
    static const std::unordered_map<std::string, std::vector<Row>> PRESET_SEQUENCES = {
        {"neutral", {
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
        }},
        {"forward", {
            {0.2f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.75f},
            {0.5f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.65f},
        }},
        {"turn", {
            {0.0f, 0.0f, 0.6f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.0f, -0.6f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
        }},
        {"sway", {
            {0.0f, 0.2f, 0.0f, 0.22f, 0.30f, 0.15f, 0.18f, -0.20f, 0.12f, 0.0f, 0.0f, 0.0f, 0.82f},
            {0.0f, 0.48f, 0.0f, 0.22f, 0.30f, 0.15f, 0.18f, -0.20f, 0.12f, 0.0f, 0.0f, 0.0f, 0.82f},
            {0.0f, -0.2f, 0.0f, 0.18f, 0.20f, 0.12f, 0.22f, -0.30f, 0.15f, 0.0f, 0.0f, 0.0f, 0.78f},
            {0.0f, -0.48f, 0.0f, 0.18f, 0.20f, 0.12f, 0.22f, -0.30f, 0.15f, 0.0f, 0.0f, 0.0f, 0.78f},
        }},
        {"combined", {
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.0f, 0.0f, 0.30f, 0.15f, 0.30f, 0.30f, -0.15f, 0.30f, 0.0f, 0.0f, 0.0f, 0.65f},
            {0.3f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
            {0.5f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.65f},
            {0.0f, 0.0f, 1.2f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.0f, -1.2f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.5f, 0.0f, 0.22f, 0.30f, 0.15f, 0.18f, -0.20f, 0.12f, 0.0f, 0.0f, 0.2f, 0.82f},
            {0.0f, -0.5f, 0.0f, 0.18f, 0.20f, 0.12f, 0.22f, -0.30f, 0.15f, 0.0f, 0.0f, -0.2f, 0.78f},
        }},
        {"tilt", {
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, -0.45f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.45f, 0.0f, 0.0f, 0.80f},
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 1.50f, 0.0f, 0.80f},
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 1.50f, 0.0f, 0.55f},
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.55f},
        }},
        {"sim2sim_test", {
            {0.0f, 0.0f, 0.0f, 0.20f, 0.25f, 0.10f, 0.20f, -0.25f, 0.10f, 0.0f, 0.0f, 0.0f, 0.75f},
        }},
    };

    std::string seq_name = params["sequence"] ? params["sequence"].as<std::string>() : std::string("neutral");
    float segment_s = params["segment_s"] ? params["segment_s"].as<float>() : 4.0f;
    float tau = params["smoothing_tau"] ? params["smoothing_tau"].as<float>() : 0.3f;

    auto it = PRESET_SEQUENCES.find(seq_name);
    const std::vector<Row>& seq = (it != PRESET_SEQUENCES.end()) ? it->second
                                                                 : PRESET_SEQUENCES.at("neutral");

    // Current segment target (advance every segment_s seconds, wrap around).
    float elapsed = static_cast<float>(env->episode_length) * env->step_dt;
    int nseg = static_cast<int>(seq.size());
    int seg = (nseg > 0 && segment_s > 1e-6f) ? (static_cast<int>(elapsed / segment_s) % nseg) : 0;
    const Row& target = seq[seg];

    // First-order low-pass smoothing. Reinit to target on (re)entry (episode_length resets to 0
    // in State_RLBase::enter -> env->reset), so it starts at the stand command, not zero.
    static std::vector<float> smoothed(13, 0.0f);
    float alpha = (tau > 1e-6f) ? std::min(1.0f, env->step_dt / tau) : 1.0f;
    if (env->episode_length <= 1)
    {
        smoothed.assign(target.begin(), target.end());
    }
    else
    {
        for (int i = 0; i < 13; ++i)
            smoothed[i] += alpha * (target[i] - smoothed[i]);
    }
    return smoothed;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{
                auto & g = env->robot->data.projected_gravity_b;
                float cos_tilt = std::clamp(-g[2], -1.0f, 1.0f);
                float tilt_deg = std::acos(cos_tilt) * 57.2958f;
                bool bad = isaaclab::mdp::bad_orientation(env.get(), 1.0);
                // Throttled heartbeat so we can watch the tilt during the transition,
                // plus a warning on the tick that actually trips the fall check.
                static int tick = 0;
                if (bad || (tick++ % 200 == 0)) {
                    spdlog::info("[WbcLowLevel] tilt={:.1f} deg  proj_g_b=[{:.3f}, {:.3f}, {:.3f}]  bad={}",
                                 tilt_deg, g[0], g[1], g[2], bad);
                }
                return bad;
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

// Mirror scripts/sim2sim.py _run_debug: same fields, same labels, same ordering, so a
// hardware run can be diffed line-for-line against sim2sim --debug-steps. Called once per
// policy step from the policy thread. Actor obs layout (wbc_lowlevel):
//   ang_vel(3) | proj_grav(3) | vel_cmd(3) | wb_cmd(10) | q(n) | dq(n) | a(n)
void State_RLBase::debug_print()
{
    if (env->last_obs.empty()) return;
    const std::vector<float>& obs = env->last_obs.begin()->second;
    std::vector<float> action = env->alg->get_action();  // raw policy output (pre scale/offset)
    const int n = static_cast<int>(action.size());
    const int prefix = static_cast<int>(obs.size()) - 3 * n;
    if (n <= 0 || prefix != 19) return;  // only the wbc_lowlevel obs layout (skips velocity)

    auto full = [](const float* p, int len) -> std::string {
        std::string s = "[";
        for (int i = 0; i < len; ++i) {
            char b[16];
            std::snprintf(b, sizeof(b), "%+.3f", p[i]);
            s += b;
            if (i + 1 < len) s += ", ";
        }
        return s + "]";
    };

    static int dstep = 0;
    std::printf("\n--- step %d ---\n", dstep++);
    std::printf("  action      : %s\n", full(action.data(), n).c_str());
    std::printf("  obs.ang_vel  : %s\n", full(obs.data() + 0, 3).c_str());
    std::printf("  obs.proj_grav: %s\n", full(obs.data() + 3, 3).c_str());
    std::printf("  obs.vel_cmd  : %s\n", full(obs.data() + 6, 3).c_str());
    std::printf("  obs.wb_cmd   : %s\n", full(obs.data() + 9, 10).c_str());
    std::printf("  obs.q        : %s\n", full(obs.data() + 19, n).c_str());
    std::printf("  obs.dq       : %s\n", full(obs.data() + 19 + n, n).c_str());
    std::printf("  obs.a        : %s\n", full(obs.data() + 19 + 2 * n, n).c_str());

    float proj_grav_z = env->robot->data.projected_gravity_b[2];
    std::printf("  root height : <n/a on hw>  proj_grav_z : %+.3f\n", proj_grav_z);

    // actuator_force equivalent: measured motor torque (tau_est), same joint order as obs.
    std::vector<float> tau(n);
    for (int i = 0; i < n; ++i)
        tau[i] = lowstate->msg_.motor_state()[env->robot->data.joint_ids_map[i]].tau_est();
    std::printf("  actuator_force: %s\n", full(tau.data(), n).c_str());
    std::fflush(stdout);
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // Diagnostic: actual motor torque (tau_est), temperature, and position tracking error
    // (commanded target q vs measured q). No software torque/joint-limit clamp exists in this
    // deploy path -- the motor firmware enforces its own limits -- so this reveals whether the
    // policy is driving joints hard or a joint is being pushed toward its mechanical limit.
    static int dbg_tick = 0;
    if (dbg_tick++ % 200 == 0) {
        float max_tau = 0.0f;   int max_tau_j = -1;
        float max_err = 0.0f;   int max_err_j = -1;
        int   max_temp = 0;     int max_temp_j = -1;
        // Firmware-reported fault state. motorstate()!=0 is an MCU error code (overload/
        // over-current/over-temp etc.); if the firmware kills a joint the reported mode()
        // also drops from the commanded/enabled value. cmd_mode is what WE told the motor.
        int   fault_j = -1;     uint32_t fault_code = 0;
        int   nfault  = 0;
        int   ankle_mode = -1;  int waist_mode = -1;  int cmd_mode = -1;
        for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
            int idx = env->robot->data.joint_ids_map[i];
            const auto & ms = lowstate->msg_.motor_state()[idx];
            float tau = ms.tau_est();
            float err = lowcmd->msg_.motor_cmd()[idx].q() - ms.q();
            int   tmp = ms.temperature()[0];
            if (std::abs(tau) > std::abs(max_tau)) { max_tau = tau; max_tau_j = idx; }
            if (std::abs(err) > std::abs(max_err)) { max_err = err; max_err_j = idx; }
            if (tmp > max_temp)                    { max_temp = tmp; max_temp_j = idx; }
            if (ms.motorstate() != 0) { ++nfault; if (fault_j < 0) { fault_j = idx; fault_code = ms.motorstate(); } }
            if (idx == 5)  ankle_mode = ms.mode();   // LeftAnkleRoll (reported)
            if (idx == 14) waist_mode = ms.mode();   // WaistPitch (reported)
        }
        cmd_mode = lowcmd->msg_.motor_cmd()[14].mode();  // what we command
        spdlog::info("[WbcLowLevel] max|tau_est|={:.1f} Nm (j{})  max|q_err|={:.3f} rad (j{})  max_temp={} C (j{})",
                     max_tau, max_tau_j, max_err, max_err_j, max_temp, max_temp_j);
        spdlog::info("[WbcLowLevel] faults={} first=j{} code=0x{:X} | reported mode: ankle(j5)={} waist(j14)={} cmd_mode={}",
                     nfault, fault_j, fault_code, ankle_mode, waist_mode, cmd_mode);
    }
}