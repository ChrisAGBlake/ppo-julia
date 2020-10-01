module PrestartEnv
export env_step, env_reset, full_reset, normalise, denormalise, get_opponent_state, load_polar, state_size, action_size, dt, max_steps
using Random
using CSV

# acceleration coefficients
const ac1 = 0.035
const ac2 = -0.009
const ac3 = 0.5
const ac4 = 0.027
const ac5 = 6.77
const ac6 = 3.68
const turn_rate_limit = deg2rad(40.0)

# polar velocity
const vmg_cwa = 0.82
const min_tws = 7.0 * 1852.0 / 3600.0
const max_tws = 21.0 * 1852.0 / 3600.0
const tws_step = 2.0 * 1852.0 / 3600.0
const v_min = 0.5
const n_tws = 8

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
const idx_tws = 23
const state_size = 23
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
const game_penalty = 0.05
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

function load_polar(fname)
    f = CSV.File(fname; header=false, delim=',', type=Float32)
    polar = Array{Float32}(undef, n_tws, 19)
    i = 1
    for row in f
        j = 1
        for col in row
            polar[i,j] = col
            j += 1
        end
        i += 1
    end
    return polar
end

function calc_polar_v(tws, cwa, polar)

    # keep cwa and tws between limits
    if cwa < 0
        cwa *= -1
    end
    if tws < min_tws
        tws = min_tws
    end
    if tws > max_tws
        tws = max_tws
    end
    
    # get the polar velocities for the tws in the table either side of the actual tws
    i = floor(Int, 1 + (tws - min_tws) / tws_step)
    if i >= n_tws
        i = n_tws - 1
    end
    j = floor(Int, 1 + cwa * 18 / π)
    if j > 18
        j = 18
    end
    
    r = (cwa - (j-1) * π / 18) / (π / 18)
    v_low = polar[i,j] * (1 - r) + polar[i,j+1] * r
    v_high = polar[i+1,j] * (1 - r) + polar[i+1,j+1] * r
    
    # interpolate between the polar velocities at the two wind speeds
    r = (tws - (min_tws + (i-1) * tws_step)) / tws_step
    v = v_low * (1 - r) + v_high * r
    
    if v < v_min
        v = v_min
    end
    return v
end

function calc_acc(tws, cwa, v, turn_rate, disp_action, polar)
    # calc polar v
    v_polar = calc_polar_v(tws, cwa, polar)
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
        acc += ac4 * ((v - ac5)^2 - ac6^2)
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

function final_dmg(state, b1, polar)
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
        acc = calc_acc(state[idx_tws], cwa, v, tr, 0, polar)

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

function calc_reward(state, prev_state, row, polar, scoring_reward::Bool)
    r = 0.0
    won = false
    pnlt = false
    ################################################################
    # check for infringement
    if overlap(state, virtual_boundary)
        if !overlap(prev_state, virtual_boundary)
            pnlt = true
            if row < 0
                if scoring_reward
                    r -= game_penalty
                else
                    r -= penalty
                end
            else
                if scoring_reward
                    r += game_penalty
                else
                    r += penalty
                end
                won = true
            end
        end

        # check for collisions
        if overlap(state, physical_boundary)
            # if collided, kill episode directly for faster iteration
            if !scoring_reward
                r -= collision_penalty
            end
            return r, pnlt, won, true
        end
    end

    ################################################################
    # check for staying in the start box
    if scoring_reward
        if state[idx_b1y] < -box_depth && prev_state[idx_b1y] >= -box_depth
            r -= game_penalty
        end
        if abs(state[idx_b1x]) > box_width && abs(prev_state[idx_b1x]) <= box_width
            r -= game_penalty
        end

        if state[idx_b2y] < -box_depth && prev_state[idx_b2y] >= -box_depth
            r += game_penalty
        end
        if abs(state[idx_b2x]) > box_width && abs(prev_state[idx_b2x]) <= box_width
            r += game_penalty
        end
    else
        if state[idx_b1y] < -box_depth
            r -= 0.01
        end
        if abs(state[idx_b1x]) > box_width
            r -= 0.01
        end
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
                r -= (start_penalty + d / 1000.0)
            else
                d = √((state[idx_b1x] - state[idx_prt_x])^2 + (state[idx_b1y] - state[idx_prt_y])^2)
                r -= (start_penalty + d / 1000.0)
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
    if !scoring_reward
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

            # add rewards for b1 distance ahead of b2 at the gun
            if state[idx_t] > prestart_duration - dt / 5
                # get distance of the boats to the line
                b1_dist = point_to_line_dist(state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y], state[idx_b1x], state[idx_b1y])
                b2_dist = point_to_line_dist(state[idx_prt_x], state[idx_prt_y], state[idx_stb_x], state[idx_stb_y], state[idx_b2x], state[idx_b2y])

                # add the reward
                r += 0.2 * (b2_dist - b1_dist) / 1000.0
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
            r += final_dmg(state, true, polar) / 1000.0
            r -= final_dmg(state, false, polar) / 1000.0
            return r, pnlt, won, true
        end

        # check for the episode ending because of time running out
        if state[idx_t] >= prestart_duration + max_t_after_start
            r += final_dmg(state, true, polar) / 1000.0
            r -= final_dmg(state, false, polar) / 1000.0
            return r, pnlt, won, true
        end
    end
    return r, pnlt, won, false
end

function env_step(state, action, row_buffer, polar, scoring_reward::Bool)
    prev_state = copy(state)

    ##################################################################
    # tws
    state[idx_tws] += (rand() - 0.5) * 0.25
    if state[idx_tws] < 3.5
        state[idx_tws] = 3.5
    end
    if state[idx_tws] > 11.5
        state[idx_tws] = 11.5
    end

    ##################################################################
    # boat 1
    # update cwa
    state[idx_b1tr] = calc_turn_rate(state[idx_b1cwa], state[idx_b1tr], action[1])
    state[idx_b1cwa] += state[idx_b1tr] * dt
    state[idx_b1cwa] = limit_pi(state[idx_b1cwa])

    # longitudinal acceleration
    acc = calc_acc(state[idx_tws], state[idx_b1cwa], state[idx_b1v], state[idx_b1tr], action[2], polar)

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
    acc = calc_acc(state[idx_tws], state[idx_b2cwa], state[idx_b2v], state[idx_b2tr], action[4], polar)    

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
    r, pnlt, won, done = calc_reward(state, prev_state, row_buffer[end], polar, scoring_reward)

    ##################################################################
    # update right of way
    row = calc_row(state)
    state[idx_row_2] = row_buffer[end]
    state[idx_row_1] = row_buffer[floor(Int, 1 / dt)]
    state[idx_row] = row
    i = size(row_buffer)[1]
    while i > 1
        row_buffer[i] = row_buffer[i-1]
        i -= 1
    end
    row_buffer[1] = row

    return r, pnlt, won, done
end

function env_reset(state, row_buffer, polar, init_states, init_idx)
    if init_idx > 100 && rand() > 0.5
        r = floor(Int, rand() * init_idx) + 1
        state[:] = view(init_states, :, r)
        for i = eachindex(row_buffer)
            row_buffer[i] = state[idx_row]
        end
        return
    end

    if rand() > 0.5
        full_reset(state, row_buffer, polar)
    else
        near_start_reset(state, row_buffer, polar)
    end
    return
end

function full_reset(state, row_buffer, polar)
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
    # tws
    state[idx_tws] = 3.5 + rand() * 8

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
    state[idx_b1v] = calc_polar_v(state[idx_tws], state[idx_b1cwa], polar)

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
    state[idx_b2v] = calc_polar_v(state[idx_tws], state[idx_b2cwa], polar)

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

function near_start_reset(state, row_buffer, polar)
    
    ##################################################################
    # marks
    length = line_length + (rand() - 0.5) * line_length_var
    skew = line_skew + (rand() - 0.5) * line_skew_var
    state[idx_prt_x] = -length * cos(skew) / 2
    state[idx_prt_y] = -length * sin(skew) / 2
    state[idx_stb_x] = length * cos(skew) / 2
    state[idx_stb_y] = length * sin(skew) / 2

    ##################################################################
    # tws
    state[idx_tws] = 3.5 + rand() * 8

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
    state[idx_b1v] = calc_polar_v(state[idx_tws], state[idx_b1cwa], polar)

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
    state[idx_b2v] = calc_polar_v(state[idx_tws], state[idx_b2cwa], polar)

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

    # tws
    state[idx_tws] -= 7.5
    state[idx_tws] /= 4
    return
end

function denormalise(state)
    # boat 1
    state[idx_b1x] *= 1000
    state[idx_b1y] *= 1000
    state[idx_b1v] /= 0.06
    state[idx_b1cwa] *= π

    # boat 2
    state[idx_b2x] *= 1000
    state[idx_b2y] *= 1000
    state[idx_b2v] /= 0.06
    state[idx_b2cwa] *= π

    # time
    state[idx_t] -= 1
    state[idx_t] *= 60
    state[idx_t] += prestart_duration
    
    # marks
    state[idx_prt_x] *= 1000
    state[idx_prt_y] *= 1000
    state[idx_stb_x] *= 1000
    state[idx_stb_y] *= 1000

    # tws
    state[idx_tws] *= 4
    state[idx_tws] += 7.5
    
    return
end

function get_opponent_state(state, opp_state)
    opp_state[1:n_boat_states] = view(state, n_boat_states+1:n_boat_states*2)
    opp_state[n_boat_states+1:n_boat_states*2] = view(state, 1:n_boat_states)
    opp_state[n_boat_states*2+1:state_size] = view(state, n_boat_states*2+1:state_size)
    opp_state[idx_row] *= -1
    opp_state[idx_row_1] *= -1
    opp_state[idx_row_2] *= -1
    return
end

end