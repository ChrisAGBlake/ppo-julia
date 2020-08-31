module CartPoleEnv
export env_step, env_reset, state_size

using Random

state_size = 4
idx_x = 1
idx_v = 2
idx_theta = 3
idx_omega = 4
gravity = 9.8
cart_mass = 1
pole_mass = 0.1
total_mass = cart_mass + pole_mass
pole_length = 0.5
pole_mass_length = pole_mass * pole_length
force_mag = 10.0
tau = 0.02
theta_threshold = deg2rad(12)
x_threshold = 2.4

function env_step(state, action)
    force = force_mag * action
    cos_theta = cos(state[idx_theta])
    sin_theta = sin(state[idx_theta])
    temp = (force + pole_mass_length * state[idx_omega]^2 * sin_theta) / total_mass
    theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length * (4.0 / 3.0 - pole_mass * cos_theta^2 / total_mass))
    acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass
    state[idx_v] += tau * acc
    state[idx_x] += tau * state[idx_v]
    state[idx_omega] += tau * theta_acc
    state[idx_theta] += tau * state[idx_omega]

    done = false
    reward = 1.0
    if abs(state[idx_x]) > x_threshold || abs(state[idx_theta]) > theta_threshold
        done = true
        reward = 0.0
    end

    return reward, done

end

function env_reset()
    # random array of state_size in the range -0.05, 0.05
    state = rand(Float32, state_size)
    state .-= 0.5
    state .*= 0.1
    return state
end

end
