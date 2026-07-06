#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>
#include <vector>
#include <array>
#include <string>
#include <algorithm>

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
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}