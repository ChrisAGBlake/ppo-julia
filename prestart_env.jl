module PrestartEnv
export env_step, env_reset, normalise, state_size, action_size, dt, max_steps
using Random

# acceleration coefficients
const ac1 = 0.035
const ac2 = -0.009
const ac3 = 0.5
const ac4 = 0.027
const ac5 = 6.77
const ac6 = 3.68
const turn_rate_limit = deg2rad(40.0)

# polar velocity coefficients
const pc1 = -0.934
const pc2 = 5.178
const pc3 = -9.64
const pc4 = 10.485
const pc5 = -26.54
const pc6 = 53.012
const pc7 = -13.634
const vmg_cwa = 0.82

# state indices
const idx_b1x = 1
const idx_b1y = 2
const idx_b1v = 3
const idx_b1cwa = 4
const idx_b1tr = 5
const idx_b1ent = 6
const idx_b1start = 7
const idx_b2x = 8
const idx_b2y = 9
const idx_b2v = 10
const idx_b2cwa = 11
const idx_b2tr = 12
const idx_b2ent = 13
const idx_b2start = 14
const idx_t = 15
const idx_prt_x = 16
const idx_prt_y = 17
const idx_stb_x = 18
const idx_stb_y = 19
const idx_row = 20
const idx_row_1 = 21
const idx_row_2 = 22
const state_size = 22
const n_boat_states = 7
const action_size = 2

# start config
const line_length = 200.0
const line_length_var = 100.0
const line_skew = 0.0
const line_skew_var = 0.5
const prestart_duration = 120.0
const max_t_after_start = 20.0
const dmg_after_start = 50.0
const box_width = 1300.0
const box_depth = 1300.0

# rewards
const penalty = 0.2
const start_penalty = 0.05
const collision_penalty = 0.05

# timestep
const dt = 1.0
const max_steps = floor(Int32, (prestart_duration + max_t_after_start) / dt)

# boat boundaries
const virtual_boundary = [[0f0, 10f0, 0f0, -10f0] [0f0, -11f0, -23f0, -11f0]]
const physical_boundary = [[0f0, 5f0, 0f0, -5f0] [0f0, -11f0, -22f0, -11f0]]

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
    if v_polar < 0.5
        v_polar = 0.5
    end
    return v_polar
end

function calc_acc(tws, cwa, v, turn_rate, disp_action)
    # calc polar v
    v_polar = calc_polar_v(cwa)
    if disp_action > 0.5
        v_polar *= 0.7
    end

    # calc acceleration based on polar speed
    delta_bsp = v_polar - v
    if delta_bsp > 10.0
        delta_bsp = 10.0
    end
    delta_cwa = abs(abs(cwa) - π / 2)
    acc = ac1 * tws * delta_bsp
    acc += ac2 * delta_bsp^2
    acc *= (1 - ac3 * delta_cwa)
    if v < ac5 + ac6 && v > ac5 - ac6
        acc += ac4 + (v - ac5)^2 - ac6^2
    end

    # add in a deceleration proportional to the turn rate
    acc -= 0.1 * v * abs(turn_rate)

    if acc < -2.0
        acc = -2.0
    end
    return acc
end

function calc_turn_rate(cwa, prev_tr, turn_angle)
    # calculate coefficients
    a = 0.2 + 1.6 / (exp(3 * turn_angle) + exp(-3 * turn_angle))
    b = abs(cwa) / deg2rad(45.0)
    if b > 1
        b = 1.0
    end
    c = b^2

    # calculate turn rate limiting to 40 deg/s
    turn_rate = a^c * turn_angle
    if turn_rate > turn_rate_limit
        turn_rate = turn_rate_limit
    end
    if turn_rate < -turn_rate_limit
        turn_rate = -turn_rate_limit
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

function line_intersection_x(x1, y1, cwa1, x2, y2, cwa2)
    m1 = 1 / tan(cwa1)
    m2 = 1 / tan(cwa2)
    c1 = y1 - m1 * x1
    c2 = y2 - m2 * x2
    x = (c1 - c2) / (m2 - m1)
    return x
end

function calc_row(state)
    flag = 0

    ################################################################
    # check if they are on different tacks
    if state[idx_b1cwa] * state[idx_b2cwa] < 0
        # starboard has right of way over port
        if state[idx_b1cwa] > 0
            return -1
        else
            return 1
        end
    else
        # they are on the same tack
        ################################################################
        # check if one boat going upwind and the other downwind
        if abs(state[idx_b1cwa]) < π / 2 && abs(state[idx_b2cwa]) > π / 2
            return 1
        end
        if abs(state[idx_b1cwa]) > π / 2 && abs(state[idx_b2cwa]) < π / 2
            return -1
        end

        ################################################################
        # they are both going upwind or both going downwind

        # get the intersection point of the lines going through the boats current travel
        x = line_intersection_x(state[idx_b1x], state[idx_b1y], state[idx_b1cwa], state[idx_b2x], state[idx_b2y], state[idx_b2cwa])

        # is the intersection point aft of both bows
        if state[idx_b1cwa] > 0
            if x < state[idx_b1x] && x < state[idx_b2x]
                flag = 1
            end
        else
            if x > state[idx_b1x] && x > state[idx_b2x]
                flag = 1
            end
        end

        # is the intersection point forward of both sterns
        if flag == 0
            s1x = state[idx_b1x] + virtual_boundary[3,2] * sin(state[idx_b1cwa])
            s2x = state[idx_b2x] + virtual_boundary[3,2] * sin(state[idx_b2cwa])
            if state[idx_b1cwa] > 0
                if x > s1x && x > s2x
                    flag = -1
                end
            else
                if x < s1x && x < s2x
                    flag = -1
                end
            end
        end
        if flag != 0
            # the intersection point is either aft of both bows or forward of both sterns
            # determine which boat is to windward at the intersection point
            # add a small amount to x opposite to the direction of travel and check which boat is further to windward at this x location
            if state[idx_b1cwa] > 0
                x += flag
            else
                x -= flag
            end
            m = 1 / tan(state[idx_b1cwa])
            b1y = state[idx_b1y] + m * (x - state[idx_b1x])
            m = 1 / tan(state[idx_b2cwa])
            b2y = state[idx_b2y] + m * (x - state[idx_b2x])
            if b2y > b1y
                # boat 2 is the windward boat
                return 1
            else
                # boat 1 is the windward boat
                return -1
            end
        end
    end
    return 0
end

function line_segment_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
    intersect = -1
    denom = (y4-y3) * (x2-x1) - (x4-x3) * (y2-y1)
    if denom == 0
        denom = 1e-6
    end
    n_a = (x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)
    n_b = (x2-x1) * (y1-y3) - (y2-y1) * (x1-x3)
    mu_a = n_a / denom
    mu_b = n_b / denom
    if mu_a >= 0 && mu_a <= 1 && mu_b >= 0 && mu_b <= 1
        intersect = mu_b
    end
    return intersect
end

function overlap(state, boundary)
    dist = √((state[idx_b2x] - state[idx_b1x])^2 + (state[idx_b2y] - state[idx_b1y])^2)
    if dist < 60
        # step through lines that make up boat 1 boundary
        for i = 1:4
            x1 = state[idx_b1x] + boundary[i,1] * cos(state[idx_b1cwa]) + boundary[i,2] * sin(state[idx_b1cwa])
            y1 = state[idx_b1y] + boundary[i,2] * cos(state[idx_b1cwa]) - boundary[i,1] * sin(state[idx_b1cwa])
            i2 = i + 1
            if i2 == 5
                i2 = 1
            end
            x2 = state[idx_b1x] + boundary[i2,1] * cos(state[idx_b1cwa]) + boundary[i2,2] * sin(state[idx_b1cwa])
            y2 = state[idx_b1y] + boundary[i2,2] * cos(state[idx_b1cwa]) - boundary[i2,1] * sin(state[idx_b1cwa])
            # step through lines that make up boat 2 boundary
            for j = 1:4
                x3 = state[idx_b2x] + boundary[j,1] * cos(state[idx_b2cwa]) + boundary[j,2] * sin(state[idx_b2cwa])
                y3 = state[idx_b2y] + boundary[j,2] * cos(state[idx_b2cwa]) - boundary[j,1] * sin(state[idx_b2cwa])
                j2 = j + 1
                if j2 == 5
                    j2 = 1
                end
                x4 = state[idx_b2x] + boundary[j2,1] * cos(state[idx_b2cwa]) + boundary[j2,2] * sin(state[idx_b2cwa])
                y4 = state[idx_b2y] + boundary[j2,2] * cos(state[idx_b2cwa]) - boundary[j2,1] * sin(state[idx_b2cwa])

                # check if the lines intersect
                if (line_segment_intersect(x1, y1, x2, y2, x3, y3, x4, y4) >= 0)
                    return true
                end
            end
        end
    end
    return false
end

function over_line(state, idx_x, idx_y)
    m = (state[idx_stb_y] - state[idx_prt_y]) / (state[idx_stb_x] - state[idx_prt_x])
    y = m * (state[idx_x] - state[idx_prt_x]) + state[idx_prt_y]
    over = state[idx_y] - y
    return over
end

function final_dmg(state, b1)
    if b1
        x = state[idx_b1x]
        y = state[idx_b1y]
        ini_y = state[idx_b1y]
        v = state[idx_b1v]
        cwa = state[idx_b1cwa]
        tr = state[idx_b1tr]
        started = state[idx_b1start]
    else
        x = state[idx_b2x]
        y = state[idx_b2y]
        ini_y = state[idx_b2y]
        v = state[idx_b2v]
        cwa = state[idx_b2cwa]
        tr = state[idx_b2tr]
        started = state[idx_b2start]
    end
    req_cwa = vmg_cwa
    if cwa < 0
        req_cwa *= -1
    end
    e = floor(Int, 5 / dt)
    for i = 1:e
        # update cwa
        turn_angle = req_cwa - cwa
        if abs(turn_angle) > 1
            turn_angle /= abs(turn_angle)
        end
        tr = calc_turn_rate(cwa, tr, turn_angle)
        cwa += tr

        # acceleration
        acc = calc_acc(6.17, cwa, v, tr, 0)

        # update velocity
        v += acc

        # update position
        y += v * cos(cwa)
    end

    # if the boat hasn't started, take away the time it will take to get to the line
    if started < 0.5
        if ini_y > 0
            y -= 2 * ini_y
        end

        if abs(x) > state[idx_stb_x]
            y -= (abs(x) - state[idx_stb_x]) * 0.6
        end
    end

    return y
end

function calc_reward(state, prev_state, row)
    r = 0.0
    won = 0
    pnlt = 0
    ################################################################
    # check for infringement
    if overlap(state, virtual_boundary)
        if !overlap(prev_state, virtual_boundary)
            pnlt = 1
            if row < 0
                r -= penalty
            else
                r += penalty
                won = 1
            end
        end

        # check for collisions
        if overlap(state, physical_boundary)
            # if collided, kill episode directly for faster iteration
            r -= collision_penalty
            return r, pnlt, won, true
        end
    end

    ################################################################
    # check for staying in the start box
    if state[idx_b1y] < -box_depth
        r -= 0.01
    end
    if abs(state[idx_b1x]) > box_width
        r -= 0.01
    end

    ################################################################
    # check for entering the start box correctly 
    # boat 1
    if state[idx_b1ent] < 0.5
        over = over_line(state, idx_b1x, idx_b1y)
        prev_over = over_line(prev_state, idx_b1x, idx_b1y)
        if over < 0 && prev_over > 0 && abs(state[idx_b1x]) < state[idx_stb_x]
            # entered correctly
            state[idx_b1ent] = 1
        end

        if state[idx_b1x] < state[idx_stb_x] && prev_state[idx_b1x] > state[idx_stb_x]
            # crossed entry mark
            if state[idx_b1y] < state[idx_stb_y]
                # didn't enter above the entry mark, penalise it
                r -= start_penalty + (state[idx_stb_y] - state[idx_b1y]) / 1000.0
                state[idx_b1ent] = 1
            end
        end

        if state[idx_b1x] > state[idx_prt_x] && prev_state[idx_b1x] < state[idx_prt_x]
            # crossed entry mark
            if state[idx_b1y] < state[idx_prt_y]
                # didn't enter above the entry mark, penalise it
                r -= start_penalty + (state[idx_prt_y] - state[idx_b1y]) / 1000.0
                state[idx_b1ent] = 1
            end
        end
            
        if state[idx_t] > 30
            # didn't enter within 30s, penalise it
            if state[idx_b1x] > 0
                d = √((state[idx_b1x] - state[idx_stb_x])^2 + (state[idx_b1y] - state[idx_stb_y])^2)
                r -= (start_penalty + d / 1000.)
            else
                d = √((state[idx_b1x] - state[idx_prt_x])^2 + (state[idx_b1y] - state[idx_prt_y])^2)
                r -= (start_penalty + d / 1000.)
            end
            state[idx_b1ent] = 1
        end
    end

    # boat 2
    if state[idx_b2ent] < 0.5
        over = over_line(state, idx_b2x, idx_b2y)
        prev_over = over_line(prev_state, idx_b2x, idx_b2y)
        if over < 0 && prev_over > 0 && abs(state[idx_b2x]) < state[idx_stb_x]
            # entered correctly
            state[idx_b2ent] = 1
        end

        if state[idx_b2x] < state[idx_stb_x] && prev_state[idx_b2x] > state[idx_stb_x]
            # crossed entry mark
            if state[idx_b2y] < state[idx_stb_y]
                # didn't enter above the entry mark, penalise it
                r += start_penalty + (state[idx_stb_y] - state[idx_b2y]) / 1000.0
                state[idx_b2ent] = 1
            end
        end

        if state[idx_b2x] > state[idx_prt_x] && prev_state[idx_b2x] < state[idx_prt_x]
            # crossed entry mark
            if state[idx_b2y] < state[idx_prt_y]
                # didn't enter above the entry mark, penalise it
                r += start_penalty + (state[idx_prt_y] - state[idx_b2y]) / 1000.0
                state[idx_b2ent] = 1
            end
        end
            
        if state[idx_t] > 30
            # didn't enter within 30s, penalise it
            if state[idx_b2x] > 0
                d = √((state[idx_b2x] - state[idx_stb_x])^2 + (state[idx_b2y] - state[idx_stb_y])^2)
                r += (start_penalty + d / 1000.)
            else
                d = √((state[idx_b2x] - state[idx_prt_x])^2 + (state[idx_b2y] - state[idx_prt_y])^2)
                r += (start_penalty + d / 1000.)
            end
            state[idx_b2ent] = 1
        end
    end

    ################################################################
    # penalise boats for crossing the line early (in the last 20s)
    if state[idx_t] > prestart_duration - 20 && state[idx_t] <= prestart_duration
        # boat 1
        over = over_line(state, idx_b1x, idx_b1y)
        if over > 0
            prev_over = over_line(prev_state, idx_b1x, idx_b1y)
            if prev_over <= 0
                if line_segment_intersect(prev_state[idx_b1x], prev_state[idx_b1y], state[idx_b1x], state[idx_b1y], state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y]) >= 0
                    # boat 1 crossed the line before the start
                    r -= start_penalty
                end
            end
        end
        
        # boat 2
        over = over_line(state, idx_b2x, idx_b2y)
        if over > 0
            prev_over = over_line(prev_state, idx_b2x, idx_b2y)
            if prev_over <= 0
                if line_segment_intersect(prev_state[idx_b2x], prev_state[idx_b2y], state[idx_b2x], state[idx_b2y], state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y]) >= 0
                    # boat 2 crossed the line before the start
                    r += start_penalty
                end
            end
        end
    end

    ################################################################
    # if the start has happened
    if state[idx_t] > prestart_duration

        # check for the boats starting
        # boat 1
        if state[idx_b1start] < 0.5
            over = over_line(state, idx_b1x, idx_b1y)
            if over > 0
                prev_over = over_line(prev_state, idx_b1x, idx_b1y)
                if prev_over <= 0
                    if line_segment_intersect(prev_state[idx_b1x], prev_state[idx_b1y], state[idx_b1x], state[idx_b1y], state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y]) >= 0
                        # started correctly
                        state[idx_b1start] = 1 
                    end
                end
            end
        end   
        # boat 2
        if state[idx_b2start] < 0.5
            over = over_line(state, idx_b2x, idx_b2y)
            if over > 0
                prev_over = over_line(prev_state, idx_b2x, idx_b2y)
                if prev_over <= 0
                    if line_segment_intersect(prev_state[idx_b2x], prev_state[idx_b2y], state[idx_b2x], state[idx_b2y], state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y]) >= 0
                        # started correctly
                        state[idx_b2start] = 1
                    end
                end
            end
        end

        # check for either boat getting to 50m dmg after the start
        if (state[idx_b1y] >= dmg_after_start && state[idx_b1start] > 0.5) || (state[idx_b2y] >= dmg_after_start && state[idx_b2start] > 0.5)
            r += final_dmg(state, true) / 1000.0
            r -= final_dmg(state, false) / 1000.0
            return r, pnlt, won, true
        end

        # check for the episode ending because of time running out
        if state[idx_t] >= prestart_duration + max_t_after_start
            r += final_dmg(state, true) / 1000.0
            r -= final_dmg(state, false) / 1000.0
            return r, pnlt, won, true
        end
    end
    return r, pnlt, won, false
end

function env_step(state, action, row_buffer)
    prev_state = copy(state)

    ##################################################################
    # boat 1
    # update cwa
    state[idx_b1tr] = calc_turn_rate(state[idx_b1cwa], state[idx_b1tr], action[1])
    state[idx_b1cwa] += state[idx_b1tr] * dt
    state[idx_b1cwa] = limit_pi(state[idx_b1cwa])

    # longitudinal acceleration
    acc = calc_acc(6.17, state[idx_b1cwa], state[idx_b1v], state[idx_b1tr], action[2])

    # update velocity
    state[idx_b1v] += acc * dt

    # update position
    state[idx_b1x] += state[idx_b1v] * sin(state[idx_b1cwa]) * dt
    state[idx_b1y] += state[idx_b1v] * cos(state[idx_b1cwa]) * dt

    ##################################################################
    # boat 2
    # update cwa
    state[idx_b2tr] = calc_turn_rate(state[idx_b2cwa], state[idx_b2tr], action[3])
    state[idx_b2cwa] += state[idx_b2tr] * dt  
    state[idx_b2cwa] = limit_pi(state[idx_b2cwa])

    # longitudinal acceleration
    acc = calc_acc(6.17, state[idx_b2cwa], state[idx_b2v], state[idx_b2tr], action[4])    

    # update velocity
    state[idx_b2v] += acc * dt

    # update position
    state[idx_b2x] += state[idx_b2v] * sin(state[idx_b2cwa]) * dt
    state[idx_b2y] += state[idx_b2v] * cos(state[idx_b2cwa]) * dt

    ##################################################################
    # time
    state[idx_t] += dt
    
    ##################################################################
    # calculate the reward for this state for boat 1
    r, pnlt, won, done = calc_reward(state, prev_state, row_buffer[end])

    ##################################################################
    # update right of way
    i = size(row_buffer)[1]
    while i > 1
        row_buffer[i] = row_buffer[i-1]
        i -= 1
    end
    row_buffer[1] = calc_row(state)

    return r, pnlt, won, done
end

function env_reset(state, row_buffer)
    if rand() > 0.5
        full_reset(state, row_buffer)
    else
        near_start_reset(state, row_buffer)
    end
    return
end

function full_reset(state, row_buffer)
    entry_side = 1

    # determine if boat 1 enters on port or starboard
    if rand() > 0.5
        entry_side = -1
    end

    ##################################################################
    # marks
    length = line_length + (rand() - 0.5) * line_length_var
    skew = line_skew + (rand() - 0.5) * line_skew_var
    state[idx_prt_x] = -length * cos(skew) / 2
    state[idx_prt_y] = -length * sin(skew) / 2
    state[idx_stb_x] = length * cos(skew) / 2
    state[idx_stb_y] = length * sin(skew) / 2

    ##################################################################
    # boat 1
    # location
    if entry_side == 1
        state[idx_b1x] = state[idx_stb_x] + 250 + rand() * 75
        state[idx_b1y] = state[idx_stb_y] + 80 + (rand() - 0.5) * 50
    else
        state[idx_b1x] = state[idx_prt_x] - 15 - rand() * 75
        state[idx_b1y] = state[idx_prt_y] + 25 + (rand() - 0.5) * 50
    end
    
    # cwa
    state[idx_b1cwa] = -deg2rad(100.0) * entry_side + (rand() - 0.5) * deg2rad(30.0)

    # turn rate
    state[idx_b1tr] = 0

    # v
    state[idx_b1v] = calc_polar_v(state[idx_b1cwa])

    # entered
    state[idx_b1ent] = 0

    # started
    state[idx_b1start] = 0

    ##################################################################
    # boat 2

    # location
    if entry_side == -1
        state[idx_b2x] = state[idx_stb_x] + 250 + rand() * 75
        state[idx_b2y] = state[idx_stb_y] + 80 + (rand() - 0.5) * 50
    else
        state[idx_b2x] = state[idx_prt_x] - 15 - rand() * 75
        state[idx_b2y] = state[idx_prt_y] + 20 + (rand() - 0.5) * 50
    end
    
    # cwa
    state[idx_b2cwa] = deg2rad(100.0) * entry_side + (rand() - 0.5) * deg2rad(30.0)

    # turn rate
    state[idx_b2tr] = 0

    # v
    state[idx_b2v] = calc_polar_v(state[idx_b2cwa])

    # entered
    state[idx_b2ent] = 0

    # started
    state[idx_b2start] = 0

    ##################################################################
    # time
    state[idx_t] = 0

    ##################################################################
    # right of way
    row = calc_row(state)
    state[idx_row] = row
    state[idx_row_1] = row
    state[idx_row_2] = row
    for i = eachindex(row_buffer)
        row_buffer[i] = row
    end
end

function near_start_reset(state, row_buffer)
    
    ##################################################################
    # marks
    length = line_length + (rand() - 0.5) * line_length_var
    skew = line_skew + (rand() - 0.5) * line_skew_var
    state[idx_prt_x] = -length * cos(skew) / 2
    state[idx_prt_y] = -length * sin(skew) / 2
    state[idx_stb_x] = length * cos(skew) / 2
    state[idx_stb_y] = length * sin(skew) / 2

    ##################################################################
    # boat 1

    # location
    state[idx_b1x] = (rand() - 0.5) * 1500
    state[idx_b1y] = (rand() - 0.8) * 800

    # cwa
    state[idx_b1cwa] = (rand() - 0.5) * 2 * pi

    # turn rate
    state[idx_b1tr] = 0

    # v
    state[idx_b1v] = calc_polar_v(state[idx_b1cwa])

    # entered
    state[idx_b1ent] = 1

    # started
    state[idx_b1start] = 0

    ##################################################################
    # boat 2

    # location, initialise close to boat 1
    state[idx_b2x] = state[idx_b1x] + (rand() - 0.5) * 500
    state[idx_b2y] = state[idx_b1y] + (rand() - 0.5) * 500

    # cwa
    state[idx_b2cwa] = (rand() - 0.5) * 2 * pi

    # turn rate
    state[idx_b2tr] = 0

    # v
    state[idx_b2v] = calc_polar_v(state[idx_b2cwa])

    # entered
    state[idx_b2ent] = 1

    # started
    state[idx_b2start] = 0

    ##################################################################
    # time, between 75 and 45 seconds to the start
    state[idx_t] = 60 + (rand() - 0.5) * 30

    ##################################################################
    # right of way
    row = calc_row(state)
    state[idx_row] = row
    state[idx_row_1] = row
    state[idx_row_2] = row
    for i = eachindex(row_buffer)
        row_buffer[i] = row
    end
end

function normalise(state)
    # boat 1
    state[idx_b1x] /= 1000
    state[idx_b1y] /= 1000
    state[idx_b1v] *= 0.06
    state[idx_b1cwa] /= π

    # boat 2
    state[idx_b2x] /= 1000
    state[idx_b2y] /= 1000
    state[idx_b2v] *= 0.06
    state[idx_b2cwa] /= π

    # time
    state[idx_t] -= prestart_duration
    state[idx_t] /= 60
    state[idx_t] += 1

    # marks
    state[idx_prt_x] /= 1000
    state[idx_prt_y] /= 1000
    state[idx_stb_x] /= 1000
    state[idx_stb_y] /= 1000
    return
end

end