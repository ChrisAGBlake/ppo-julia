include("prestart_env.jl")
include("utils.jl")
using .PrestartEnv
using .Utils
using Flux
using Statistics: mean, sum
using Dates: now
using Distributions
using CUDA
using BSON: @save, @load

function act(model, log_std, state, c)
    σ = exp.(log_std)
    μ = model(state)
    value = μ[end]
    μ = view(μ, 1:action_size)

    # sample an action from a normal distribution with mean μ and standard deviation σ
    d = Normal.(μ, σ)
    action = rand.(d)

    # calculate the log probability of this action
    v = σ.^2
    log_prob = sum(-((action .- μ).^2) ./ (2 .* v) .- log_std .- c)

    return action, log_prob, value
end

function run_episode(model, log_std, opponent, polar, states, actions, values, combined_actions, log_probs, rewards, state, norm_state, opp_state, row_buffer, init_states, init_idx, c)
    
    env_reset(state, row_buffer, polar, init_states, init_idx)
    n_steps = max_steps
    for i = 1:max_steps
        # get the normalised state
        norm_state[:] = state
        normalise(norm_state)

        # add state to buffer
        states[:,i] = norm_state

        # get action, log prob of action and value estimate
        action, log_prob, value = act(model, log_std, norm_state, c)
        actions[:,i] = action
        values[i] = value
        log_probs[i] = log_prob
        combined_actions[1:2] = action

        # get the opponents actions
        if opponent === nothing
            combined_actions[3] = rand() * 2 - 1
            combined_actions[4] = rand()
        else
            get_opponent_state(norm_state, opp_state)
            opponent_action = opponent(opp_state)
            combined_actions[3:4] = view(opponent_action, 1:action_size)
        end

        # update the environment
        reward, pnlt, won, done = env_step(state, combined_actions, row_buffer, polar, false)

        # if there has been a penalty, add the state 10s before to the init_states array
        if pnlt && i > 10 / dt
            n = floor(Int, i - 10 / dt)
            init_idx += 1
            init_states[:, init_idx] = view(states, :, n)
            denormalise(view(init_states, :, init_idx))
            if init_idx == 10000
                init_states[:,1:1000] = view(init_states, :,9001:10000)
                init_idx = 1000
            end
        end

        # add reward to buffer
        rewards[i] = reward

        # check for episode ending
        if done
            n_steps = i
            break
        end
    end
    
    # set the final value to 0
    n_steps += 1
    values[n_steps] = 0

    return n_steps, init_idx
end

function discount_cumsum(values, discount_array)
    sz = size(values)[1]
    res = similar(values)
    for i = 1:sz
        res[i] = sum(view(values, i:sz) .* view(discount_array, 1:sz-i+1))
    end
    return res
end

function loss(model, log_std, states, actions, adv_est, log_prob_old, rewards2go, values_old, hp)
    
    # get model action and value predictions from the states
    μ = model(states)
    values = view(μ, action_size+1, :)
    μ = view(μ, 1:action_size, :)

    # calculate policy loss
    σ = exp.(log_std)  
    v = σ.^2
    log_prob = view(sum(-((actions .- μ).^2) ./ (2 .* v) .- log_std .- hp.c, dims=1), 1, :)
    ratio = exp.(log_prob .- log_prob_old)
    clip_adv = clamp.(ratio, 1-hp.ϵ, 1+hp.ϵ) .* adv_est
    p_loss = -mean(min.(ratio .* adv_est, clip_adv))

    # calculate entropy
    entropy = sum(log_std .+ hp.d)
    entropy *= hp.ent_coef

    # calculate clipped value loss
    v_clip = values_old .+ clamp.(values .- values_old, -hp.ϵ, hp.ϵ)
    v_loss1 = (values .- rewards2go) .^ 2
    v_loss2 = (v_clip .- rewards2go) .^ 2
    v_loss = hp.vf_coef * mean(max.(v_loss1, v_loss2))

    return p_loss + v_loss - entropy

end

function setup_optimisers(lr)
    opt = ADAM(lr) |> gpu
    return opt
end

function setup_model(lr, large, model_n)
    # load the latest checkpoint or start from scratch if it doesn't exist
    idx = get_checkpoint_idx(model_n)
    if idx === nothing
        # no checkpoints present, create the model critic model
        if large
            n1 = 256
            n2 = 512
            # create the network
            cpu_model = Chain(
                Dense(state_size, n1, tanh),
                Dense(n1, n2, tanh),
                Dense(n2, n2, tanh),
                Dense(n2, n1, tanh),
                Dense(n1, action_size + 1)
            )
        else
            n1 = 128
            # create the network
            cpu_model = Chain(
                Dense(state_size, n1, tanh),
                Dense(n1, n1, tanh),
                Dense(n1, n1, tanh),
                Dense(n1, action_size + 1)
            )
        end

        # create the log standard deviations for the model
        log_std = Array{Float32}(undef, action_size)
        for i = eachindex(log_std)
            log_std[i] = -0.5
        end
        idx = 0
    else
        base_dir = string(pwd(), "/models/")
        println("starting training from checkpoint ", idx)
        @load string(base_dir, "model", model_n, "_", idx, ".bson") cpu_model log_std
    end

    # shift to the gpu
    model = cpu_model |> gpu

    # setup the optimisers
    opt = setup_optimisers(lr)

    return model, cpu_model, log_std, opt, idx
end

function update_ac(model, log_std, opt, states, actions, adv, log_probs, r2g, values, hp)

    # update the network
    ps = Flux.params(model, log_std)
    for j = 1:hp.n_updates
        g = gradient(() -> loss(model, log_std, states, actions, adv, log_probs, r2g, values, hp), ps)
        Flux.update!(opt, ps, g)
    end

    return
end

function get_checkpoint_idx(model_n)
    
    # get checkpoints directory 
    base_dir = string(pwd(), "/models/")

    # get the array of checkpoints
    checkpoints = readdir(base_dir)

    if size(checkpoints)[1] > 0
        # get the highest index of the checkpoints
        name = string("model", model_n)
        r = Regex("$name")
        idx = 1
        for i = eachindex(checkpoints)
            m = match(r, checkpoints[i])
            if m !== nothing
                m = match(r"_\d+", checkpoints[i]).match
                n = parse(Int, match(r"\d", m).match)
                if n > idx
                    idx = n
                end
            end
        end
        return idx
    else
        return nothing
    end
end

function choose_opponent()

    # get checkpoints directory 
    base_dir = string(pwd(), "/models/")

    # get the potential opponents
    checkpoints = readdir(base_dir)
    sz = size(checkpoints)[1]

    if sz > 0
        # select an opponent
        r = floor(Int32, rand() * sz + 1)
        
        # load the opponent
        @load string(base_dir, checkpoints[r]) cpu_model log_std
    else
        # no models present yet, create one at random
        cpu_model = nothing
    end

    return cpu_model
end

function get_batch(model, log_std, gamma_arr, gamma_lam_arr, polar, hp, init_states, init_idx)

    # setup data buffers
    states_buf = Array{Float32}(undef, state_size, hp.batch_size)
    actions_buf = Array{Float32}(undef, action_size, hp.batch_size)
    log_probs_buf = Array{Float32}(undef, hp.batch_size)
    r2g_buf = Array{Float32}(undef, hp.batch_size)
    adv_buf = Array{Float32}(undef, hp.batch_size)
    val_buf = Array{Float32}(undef, hp.batch_size)

    # select an opponent
    opponent = choose_opponent()

    # run a batch of episodes in parallel
    n_threads = Threads.nthreads()
    batch_size = floor(Int, hp.batch_size / n_threads)
    Threads.@threads for i = 1:n_threads
        states = Array{Float32}(undef, state_size, max_steps)
        actions = Array{Float32}(undef, action_size, max_steps)
        values = Array{Float32}(undef, max_steps+1)
        combined_actions = Array{Float32}(undef, action_size * 2)
        log_probs = Array{Float32}(undef, max_steps)
        rewards = Array{Float32}(undef, max_steps)
        state = Array{Float32}(undef, state_size)
        norm_state = Array{Float32}(undef, state_size)
        opp_state = Array{Float32}(undef, state_size)
        row_buffer = Array{Float32}(undef, floor(Int, 2 / dt))
        n = 0
        ts = (i - 1) * batch_size
        while n < batch_size

            # run an episode
            sz, init_idx = run_episode(model, log_std, opponent, polar, states, actions, values, combined_actions, log_probs, rewards, state, norm_state, opp_state, row_buffer, init_states, init_idx, hp.c)

            # calculate the rewards to go
            r2g = discount_cumsum(view(rewards, 1:sz-1), gamma_arr)

            # calculate the advantage estimates
            δ = view(rewards, 1:sz-1) + hp.γ * view(values, 2:sz) - view(values, 1:sz-1)
            adv_est = discount_cumsum(δ, gamma_lam_arr)

            # update the buffers
            s = n + 1
            sz = min(sz-1, batch_size - n)
            n += sz
            states_buf[:, ts+s:ts+n] = view(states, :, 1:sz)
            actions_buf[:, ts+s:ts+n] = view(actions, :, 1:sz)
            log_probs_buf[ts+s:ts+n] = view(log_probs, 1:sz)
            r2g_buf[ts+s:ts+n] = view(r2g, 1:sz)
            adv_buf[ts+s:ts+n] = view(adv_est, 1:sz)
            val_buf[ts+s:ts+n] = view(values, 1:sz)
            
        end
    end

    # normalise the advantage estimates
    μ = mean(adv_buf)
    σ = std(adv_buf)
    adv_buf = (adv_buf .- μ) ./ σ

    #shift to the gpu
    gpu_log_std = log_std |> gpu
    gpu_states_buf = states_buf |> gpu
    gpu_actions_buf = actions_buf |> gpu
    gpu_adv_buf = adv_buf |> gpu
    gpu_log_probs_buf = log_probs_buf |> gpu
    gpu_r2g_buf = r2g_buf |> gpu
    gpu_val_buf = val_buf |> gpu

    return gpu_log_std, gpu_states_buf, gpu_actions_buf, gpu_adv_buf, gpu_log_probs_buf, gpu_r2g_buf, gpu_val_buf, init_idx
end

function train(hp, exp_name, s3, model_n)

    # get checkpoints directory, create if it doesn't exist
    base_dir = string(pwd(), "/models/")
    try
        mkdir(base_dir)
    catch
    end

    # sync with s3
    n = sync_with_s3(s3, base_dir, exp_name)

    # setup arrays to speed up calculation of advantage estimates
    gamma_arr = Array{Float32}(undef, max_steps)
    gamma_lam_arr = Array{Float32}(undef, max_steps)
    for i = eachindex(gamma_arr)
        gamma_arr[i] = hp.γ^(i-1)
        gamma_lam_arr[i] = (hp.γ * hp.λ)^(i-1)
    end

    # load the latest checkpoint or start from scratch if it doesn't exist
    model, cpu_model, log_std, opt, idx = setup_model(hp.lr, hp.large_network, model_n)

    # load the polar
    polar = load_polar("env/polar.csv")

    # initialise the array of problematic reset states
    init_states = Array{Float32}(undef, state_size, 10000)
    init_idx = 0

    n_epochs = floor(Int, 1e7 / hp.batch_size)
    while true
        # run n_epochs updates
        for i = 1:n_epochs
            st = now()

            # get a batch of data to train on
            gpu_log_std, states, actions, adv, log_probs, r2g, values, init_idx = get_batch(cpu_model, log_std, gamma_arr, gamma_lam_arr, polar, hp, init_states, init_idx)
            
            # update the model critic networks
            update_ac(model, gpu_log_std, opt, states, actions, adv, log_probs, r2g, values, hp)
            
            # shift back to the cpu
            cpu_model = model |> cpu
            log_std = gpu_log_std |> cpu

            # timing for performance
            print(i, " time: ", now() - st, "\r")
            
        end

        # checkpoint the model
        idx += 1
        model_name = string(base_dir, "model", model_n, "_", idx, ".bson")
        @save model_name cpu_model log_std
        s3.upload_file(model_name, s3_bucket, string(exp_name, "/models/model", model_n, "_", idx, ".bson"))

        # sync with s3 to get checkpoints from other instances
        n = sync_with_s3(s3, base_dir, exp_name)
    end
end

# set the hyper parameters for the training
struct hyper_parameters
    γ::Float32
    λ::Float32
    ϵ::Float32
    c::Float32
    lr::Float32
    ent_coef::Float32
    vf_coef::Float32
    d::Float32
    n_updates::Int32
    batch_size::Int32
    large_network::Bool
end
hp = hyper_parameters(0.99, 0.95, 0.2, log(sqrt(2 * π)), 3e-4, 0.01, 0.25, 0.5 * log(2 * π * ℯ), 10, 100000, true)

# initialise aws setup
s3 = connect_to_s3()

# run the training
exp_name = ARGS[1]
model_n = ARGS[2]
train(hp, exp_name, s3, model_n)
