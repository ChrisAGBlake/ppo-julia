module PrestartTrajectoryEnv
export env_step, env_reset, normalise, state_size, action_size
using Random

# acceleration coefficients
const ac1 = 0.035
const ac2 = -0.009
const ac3 = 0.5
const ac4 = 0.027
const ac5 = 6.77
const ac6 = 3.68

# polar velocity coefficients
const pc1 = -0.934
const pc2 = 5.178
const pc3 = -9.64
const pc4 = 10.485
const pc5 = -26.54
const pc6 = 53.012
const pc7 = -13.634

# state indices
const idx_t = 1
const idx_tws = 2
const idx_x = 3
const idx_y = 4
const idx_v = 5
const idx_cwa = 6
const idx_tr = 7
const state_size = 8
const action_size = 1

# timestep
const dt = 2.0

function limit_pi(val)
    if val > π
        val -= 2 * π
    end
    if val < -π
        val += 2 * π
    end
    return val
end

function calc_polar_v(cwa)
    if cwa < 0
        cwa *= -1
    end
    v_polar = pc1 * cwa^6 + pc2 * cwa^5 + pc3 * cwa^4 + pc4 * cwa^3 + pc5 * cwa^2 + pc6 * cwa + pc7
    if v_polar < 2
        v_polar = 2
    end
    return v_polar
end

function calc_acc(tws, cwa, v, turn_rate)
    # calc polar v
    v_polar = calc_polar_v(cwa)

    # calc acceleration based on polar speed
    delta_bsp = v_polar - v
    delta_cwa = abs(abs(cwa) - π / 2)
    acc = ac1 * tws * delta_bsp
    acc += ac2 * delta_bsp^2
    acc *= (1 - ac3 * delta_cwa)
    if v < ac5 + ac6 && v > ac5 - ac6
        acc += ac4 + (v - ac5)^2 - ac6^2
    end

    # add in a deceleration proportional to the turn rate
    acc -= 0.1 * v * abs(turn_rate)
    return acc
end

function calc_turn_rate(cwa, prev_tr, action)
    # calculate coefficients
    a = 0.2 + 1.6 / (exp(3 * action) + exp(-3 * action))
    b = abs(cwa) / deg2rad(45)
    if b > 1
        b = 1
    end
    c = b^2

    # calculate turn rate limiting to 40 deg/s
    turn_rate = a^c * action
    if turn_rate > deg2rad(40)
        turn_rate = deg2rad(40)
    end
    if turn_rate < deg2rad(-40)
        turn_rate = deg2rad(-40)
    end

    # limit further based on previous turn rate
    turn_limit = deg2rad(2 + (dt / 0.05 - 1) * 2)
    if turn_rate > prev_tr + turn_limit
        turn_rate = prev_tr + turn_limit
    end
    if turn_rate < prev_tr - turn_limit
        turn_rate = prev_tr - turn_limit
    end

    return turn_rate
end

function point_to_line_dist(x1, y1, x2, y2, x3, y3)
    mu = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1)^2 + (y2 - y1)^2)
    if mu < 0
        mu = 0
    end
    if mu > 1
        mu = 1
    end

    x4 = x1 + mu * (x2 - x1)
    y4 = y1 + mu * (y2 - y1)
    d = √((x4 - x3)^2 + (y4 - y3)^2)
    return d
end

function calc_reward(state, prev_state)
    d = point_to_line_dist(state[idx_x], state[idx_y], prev_state[idx_x], prev_state[idx_y], 0, 0)

    if d < 30
        r = 1 - abs(state[idx_cwa] - deg2rad(90)) / 3
        return r, true
    end

    if state[idx_t] > 120
        r = -d / 1000  - abs(state[idx_cwa] - deg2rad(90)) / 3
        return r, true
    end

    return -0.01, false
end

function env_step(state, action)
    prev_state = copy(state)

    # update turn rate
    state[idx_tr] = calc_turn_rate(state[idx_cwa], state[idx_tr], action[1])

    # update cwa
    state[idx_cwa] += state[idx_tr] * dt
    state[idx_cwa] = limit_pi(state[idx_cwa])

    # update velocity
    acc = calc_acc(state[idx_tws], state[idx_cwa], state[idx_v], state[idx_tr])
    state[idx_v] += acc * dt
    if state[idx_v] < 2
        state[idx_v] = 2
    end

    # update position
    state[idx_x] += state[idx_v] * sin(state[idx_cwa]) * dt
    state[idx_y] += state[idx_v] * cos(state[idx_cwa]) * dt

    # update time
    state[idx_t] += dt

    # calc reward
    reward, done = calc_reward(state, prev_state)

    return reward, done
end

function env_reset(state)
    
    # time
    state[idx_t] = 0

    # tws
    state[idx_tws] = rand() * 7 + 3.5

    # location
    state[idx_x] = (rand() - 0.5) * 2000
    state[idx_y] = (rand() - 0.5) * 2000

    # v
    state[idx_v] = rand() * 15 + 5

    # cwa
    state[idx_cwa] = (rand() - 0.5) * 2 * π

    # turn rate
    state[idx_tr] = 0

    return
end

function normalise(state)
    # time
    state[idx_t] /= 60
    state[idx_t] -= 1

    # tws
    state[idx_tws] /= 5
    state[idx_tws] -= 1

    # location
    state[idx_x] /= 1000
    state[idx_y] /= 1000

    # speed
    state[idx_v] /= 10
    state[idx_v] -= 1

    # cwa
    state[idx_cwa] /= π

    return
end

end