include("prestart_env.jl")
include("utils.jl")
using .PrestartEnv
using .Utils
using Flux
using BSON: @load
using PyCall

sb = pyimport("stable_baselines")

function prestart(model1, model2, polar)

    # initialise the episode
    state = Array{Float32}(undef, state_size)
    norm_state = Array{Float32}(undef, state_size)
    opp_state = Array{Float32}(undef, state_size)
    row_buffer = Array{Float32}(undef, floor(Int, 2 / dt))
    actions = Array{Float32}(undef, action_size * 2)
    full_reset(state, row_buffer, polar)
    rewards = 0
    n_penalties = 0
    n_won = 0
    b1_action = nothing
    b2_action = nothing

    # run the episode
    while true
        # get the normalised state
        norm_state[:] = state
        normalise(norm_state)

        # get the boat 1 action
        try
            b1_action = model1(norm_state)
            b1_action = view(b1_action, 1:action_size)
        catch
            b1_action, n = model1.predict(norm_state, deterministic=true)
        end

        # get the boat 2 action
        get_opponent_state(norm_state, opp_state)
        try
            b2_action = model2(opp_state)
            b2_action = view(b2_action, 1:action_size)
        catch
            b2_action, n = model2.predict(opp_state, deterministic=true)
        end

        # calculate the combined actions
        actions[1:2] = b1_action
        actions[3:4] = b2_action

        # apply these actions to transition the state
        reward, pnlt, won, done = env_step(state, actions, row_buffer, polar, true)
        rewards += reward
        if pnlt
            n_penalties += 1
        end
        if won
            n_won += 1
        end
        if done
            break
        end
    end
    win = true
    if rewards < 0
        win = false
    end
    return win, n_penalties, n_won
end

function download_models(exp_names)
    s3 = connect_to_s3()
    n = 0
    for i = eachindex(exp_names)
        dir = string(pwd(), "/historical_models/", exp_names[i], "/")
        try
            mkdir(dir)
        catch
        end
        en = sync_with_s3(s3, dir, exp_names[i])
        if en > n
            n = en
        end
    end
    return n
end

function score_models(exp_names, checkpoints, nr)
    
    # load the polar
    polar = load_polar("env/polar.csv")

    # get the model filenames to score
    filenames = Array{String}(undef, 10000)
    sz = 0
    for i = eachindex(exp_names)
        dir = string(pwd(), "/historical_models/", exp_names[i], "/")
        names = readdir(dir, join=true)
        for j = eachindex(names)
            if match(r"bson", names[j]) !== nothing
                s = match(r"_\d+[.]", names[j]).match
                step = parse(Int, match(r"\d+", s).match)
                for k = eachindex(checkpoints)
                    if step == checkpoints[k]
                        sz += 1
                        filenames[sz] = names[j]
                        break
                    end
                end
            end
            if match(r"dat", names[j]) !== nothing
                s = match(r"_\d+[.]", names[j]).match
                step = parse(Int, match(r"\d+", s).match)
                for k = eachindex(checkpoints)
                    if step == checkpoints[k]
                        sz += 1
                        filenames[sz] = names[j]
                        break
                    end
                end
            end
        end
    end
    filenames = view(filenames, 1:sz)

    # load the models
    models = Array{Any}(undef, 0)
    for i = 1:sz
        if match(r"bson", filenames[i]) !== nothing
            @load filenames[i] cpu_model log_std
            models = cat(models, cpu_model, dims=1)
        end
        if match(r"dat", filenames[i]) !== nothing
            model = sb.PPO2.load(filenames[i])
            models = cat(models, model, dims=1)
        end
    end

    # run the episodes
    scores = zeros(Float32, sz)
    n_won_penalties = zeros(Float32, sz)
    n_lost_penalties = zeros(Float32, sz)
    for n = 1:nr
        println("repeat ", n, " of ", nr)
        for i = 1:sz
            print(i, " of ", sz, "\r")
            for j = 1:sz
                if i != j
                    # run the episode
                    win, np, nw = prestart(models[i], models[j], polar)
                    if win
                        scores[i] += 1
                    else
                        scores[j] += 1
                    end
                    n_won_penalties[i] += nw
                    n_lost_penalties[i] += np - nw
                    n_won_penalties[j] += np - nw
                    n_lost_penalties[j] += nw

                end
            end
        end
    end

    # normalise the scores
    for i = eachindex(scores)
        scores[i] /= sz * 2 - 2
        scores[i] /= nr
    end

    # write out the scores in a csv file
    open("scores.csv", "w") do io
        write(io,"filename,score\n")
        for i = eachindex(filenames)
            write(io, string(filenames[i], ",", scores[i], "\n"))
        end
    end

end

exp_names = ["tws-orig", "tws-compare", "tws-compare-5models", "jl-multi", "jl-tanh"]
n = download_models(exp_names)
checkpoints = [0,1,2,3,5,7,9]
if n <= 50
    checkpoints = cat(checkpoints, collect(Int64, 12:3:30), dims=1)
    checkpoints = cat(checkpoints, collect(Int64, 35:5:50), dims=1)
else
    checkpoints = collect(Int64, 0:5:300)
end
score_models(exp_names, checkpoints, 1)
