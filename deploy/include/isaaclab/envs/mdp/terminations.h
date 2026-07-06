#pragma once

#include <algorithm>
#include <cmath>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

// Fall protection: projected_gravity_b is gravity in the pelvis frame, so
// acos(-projected_gravity_b.z) is the pelvis tilt from vertical (0 = upright).
// Returns true when tilt exceeds limit_angle (default 1.0 rad ~= 57 deg) -> FSM drops to Passive.
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    float cos_tilt = std::clamp(-data[2], -1.0f, 1.0f);
    return std::acos(cos_tilt) > limit_angle;
}

} 
} 