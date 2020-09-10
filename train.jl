include("./prestart_env.jl")
using .PrestartEnv
using Flux, Distributions
using Flux: params, update!
using Statistics: mean
using Dates: now
using CUDA
using BSON: @save, @load

function act(actor, act_log_std, state)
    σ = exp.(act_log_std)
    μ = actor(state)

    # sample an action from a normal distribution with mean μ and standard deviation σ
    d = Normal.(μ, σ)
    action = rand.(d)

    # calculate the log probability of this action
    v = σ.^2
    log_scale = log.(σ)
    log_prob = -((action .- μ).^2) ./ (2 .* v) .- log_scale .- log(sqrt(2 * π))

    return action, log_prob
end

function run_episode(actor, act_log_std, critic, opponent, use_gpu::Bool)
    states = Array{Float32}(undef, state_size, max_steps+1)
    actions = Array{Float32}(undef, action_size, max_steps)
    combined_actions = Array{Float32}(undef, action_size * 2)
    log_probs = Array{Float32}(undef, action_size, max_steps)
    rewards = Array{Float32}(undef, max_steps + 1)
    state = Array{Float32}(undef, state_size)
    norm_state = Array{Float32}(undef, state_size)
    row_buffer = Array{Float32}(undef, floor(Int, 2 / dt))
    env_reset(state, row_buffer)
    
    n_steps = max_steps
    for i = 1:max_steps
        # get the normalised state
        norm_state[:] = state[:]
        normalise(norm_state)

        # add state to buffer
        states[:,i] = norm_state[:]

        # get action, log prob of action and value estimate
        action, log_prob = act(actor, act_log_std, norm_state)
        actions[:,i] = action[:]
        log_probs[:,i] = log_prob[:]

        # get the opponents actions
        for j = eachindex(action)
            combined_actions[j] = action[j]
        end
        combined_actions[3] = rand() * 2 - 1
        combined_actions[4] = rand()

        # update the environment
        reward, pnlt, won, done = env_step(state, combined_actions, row_buffer)

        # add reward to buffer
        rewards[i] = reward

        # check for episode ending
        if done
            n_steps = i
            break
        end
    end
    if use_gpu
        values = critic(view(states |> gpu, :, 1:n_steps+1))
        values = values |> cpu
    else
        values = critic(view(states, :, 1:n_steps+1))
    end
    # set the final value to 0
    values[n_steps + 1] = 0
    return view(states, :, 1:n_steps), view(actions, :, 1:n_steps), view(log_probs, :, 1:n_steps), view(rewards, 1:n_steps), view(values, 1:n_steps + 1)
end

function discount_cumsum(values, discount_array)
    sz = size(values)[1]
    res = similar(values)
    for i = 1:sz
        res[i] = sum(view(values, i:sz) .* view(discount_array, 1:sz-i+1))
    end
    return res
end

function policy_loss(actor, act_log_std, states, actions, adv_est, log_prob_old, ϵ, c)
    μ = actor(states)
    σ = exp.(act_log_std)  
    v = σ.^2
    log_scale = log.(σ)
    log_prob = -((actions .- μ).^2) ./ (2 .* v) .- log_scale .- c
    ratio = exp.(log_prob .- log_prob_old)
    clip_adv = clamp.(ratio, 1-ϵ, 1+ϵ) .* adv_est'
    loss = -mean(min.(ratio .* adv_est', clip_adv))
    return loss
end

function value_loss(critic, states, rewards2go)
    values = critic(states)
    loss = (values' .- rewards2go) .^ 2
    return mean(loss)
end

function setup_model(lr, use_gpu::Bool)
    # create the policy network and optimiser
    actor = Chain(
        Dense(state_size, 256, relu),
        Dense(256, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 256, relu),
        Dense(256, action_size)
    )
    act_log_std = Array{Float32}(undef, action_size)
    for i = eachindex(act_log_std)
        act_log_std[i] = -0.5
    end
    act_optimiser = ADAM(lr)
    if use_gpu
        act_optimiser = act_optimiser |> gpu
    end

    # create the value network and optimiser
    critic = Chain(
        Dense(state_size, 256, relu),
        Dense(256, 512, relu),
        Dense(512, 512, relu),
        Dense(512, 256, relu),
        Dense(256, 1)
    )
    crt_optimiser = ADAM(lr)
    if use_gpu
        critic = critic |> gpu
        act_optimiser = act_optimiser |> gpu
    end

    return actor, act_log_std, act_optimiser, critic, crt_optimiser
end

function update_ac(actor, act_log_std, act_optimiser, critic, crt_optimiser, states, actions, adv, log_probs, r2g, hp)
    # update the policy network
    st = now()
    act_params = params(actor, act_log_std)
    for j = 1:hp.n_p_updates
        p_grad = gradient(() -> policy_loss(actor, act_log_std, states, actions, adv, log_probs, hp.ϵ, hp.c), act_params)
        update!(act_optimiser, act_params, p_grad)
    end
    println("time to update policy: ", now() - st)
    
    # update the value network
    st = now()
    crt_params = params(critic)
    for j = 1:hp.n_v_updates
        v_grad = gradient(() -> value_loss(critic, states, r2g), crt_params)
        update!(crt_optimiser, crt_params, v_grad)
    end
    println("time to update value function: ", now() - st)
    println()
    return
end

function get_checkpoint_idx()
    # get the array of checkpoints
    checkpoints = readdir("checkpoints/")

    # get the highest index of the checkpoints
    idx = 1
    checkpoint_names = Array{String}(undef, size(checkpoints)[1])
    for i = eachindex(checkpoints)
        m = match(r"actor", checkpoints[i])
        if m != nothing
            n = parse(Int, match(r"\d+", checkpoints[i]).match)
            if n > idx
                idx = n
            end
    end
    return idx
end

function choose_opponent()

    # get the potential opponents
    checkpoints = readdir("checkpoints/")
    j = 1
    checkpoint_names = Array{String}(undef, size(checkpoints)[1])
    for i = eachindex(checkpoints)
        m = match(r"actor", checkpoints[i])
        if m != nothing
            checkpoint_names[j] = checkpoints[i]
            j += 1
        end
    end
    j -= 1

    # select an opponent
    r = floor(Int32, rand() * j + 1)
    
    # load the opponent
    @load checkpoint_names[r] opponent

    return opponent
end

function train(hp, use_gpu::Bool)

    # initialise results file
    write("training_rewards.csv", "ave_episode_reward\n")

    # setup arrays to speed up calculation of advantage estimates
    gamma_arr = Array{Float32}(undef, max_steps)
    gamma_lam_arr = Array{Float32}(undef, max_steps)
    for i = eachindex(gamma_arr)
        gamma_arr[i] = hp.γ^(i-1)
        gamma_lam_arr[i] = (hp.γ * hp.λ)^(i-1)
    end

    # create the actor critic model
    actor, act_log_std, act_optimiser, critic, crt_optimiser = setup_model(hp.lr, use_gpu)

    idx = 1
    while true
        # run n_epochs updates
        for i = 1:hp.n_epochs
            st = now()

            # setup data buffers
            n = 0
            states_buf = Array{Float32}(undef, state_size, 0)
            actions_buf = Array{Float32}(undef, action_size, 0)
            log_probs_buf = Array{Float32}(undef, action_size, 0)
            r2g_buf = Array{Float32}(undef, 0)
            adv_buf = Array{Float32}(undef, 0)
            sr = 0.0
            ne = 0.0

            # select an opponent
            opponent = choose_opponent()

            # run a batch of episodes
            while n < hp.batch_size

                # run an episode
                states, actions, log_probs, rewards, values = run_episode(actor, act_log_std, critic, opponent, use_gpu)
                sr += sum(rewards)
                ne += 1

                # calculate the rewards to go
                r2g = discount_cumsum(rewards, gamma_arr)

                # calculate the advantage estimates
                sz = size(values)[1]
                δ = view(rewards, 1:sz-1) + hp.γ * view(values, 2:sz) - view(values, 1:sz-1)
                adv_est = discount_cumsum(δ, gamma_lam_arr)

                # update the buffers
                states_buf = cat(states_buf, states, dims=2)
                actions_buf = cat(actions_buf, actions, dims=2)
                log_probs_buf = cat(log_probs_buf, log_probs, dims=2)
                r2g_buf = cat(r2g_buf, r2g, dims=1)
                adv_buf = cat(adv_buf, adv_est, dims=1)
                n = size(states_buf)[end]
                
            end

            sr /= ne
            println("time to run episodes: ", now() - st)
            println(i, ", ave rewards per episode: ", sr)
            open("training_rewards.csv", "a") do io
                write(io, string(sr, "\n"))
            end

            # normalise the advantage estimates
            μ = mean(adv_buf)
            σ = std(adv_buf)
            adv_buf = (adv_buf .- μ) ./ σ

            #shift to the gpu
            if use_gpu
                actor = actor |> gpu
                act_log_std = act_log_std |> gpu
                states_buf = states_buf |> gpu
                actions_buf = actions_buf |> gpu
                adv_buf = adv_buf |> gpu
                log_probs_buf = log_probs_buf |> gpu
                r2g_buf = r2g_buf |> gpu
            end
            
            # update the actor critic networks
            update_ac(actor, act_log_std, act_optimiser, critic, crt_optimiser, states_buf, actions_buf, adv_buf, log_probs_buf, r2g_buf, hp)

            # shift back to the cpu
            if use_gpu
                actor = actor |> cpu
                act_log_std = act_log_std |> cpu
            end
            
        end

        # checkpoint the model
        @save string("checkpoints/actor_", idx, ".bson") actor
        @save string("checkpoints/act_log_std_", idx, ".bson") act_log_std
        @save string("checkpoints/critic_", idx, ".bson") critic
        idx += 1
    end
end

# set the hyper parameters for the training
struct hyper_parameters
    γ::Float32
    λ::Float32
    ϵ::Float32
    c::Float32
    lr::Float32
    n_p_updates
    n_v_updates
    batch_size
    n_epochs
end
hp = hyper_parameters(0.99, 0.97, 0.2, log(sqrt(2 * π)), 3e-4, 10, 10, 10000, 10)

# run the training
st = now()
train(hp, true)
println("total time: ", now() - st)
