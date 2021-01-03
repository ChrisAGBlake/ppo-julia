module PPO

using Flux
using Statistics: mean, sum
using Dates: now
using Distributions

const state_size = 20
const action_size = 2
const max_steps = 150

function env_reset(state)
    for i in eachindex(state)
        state[i] = rand()
    end
    return
end

function env_step(state, action)
    for i in eachindex(state)
        state[i] = rand()
    end
    return rand() + action[1], false
end

function discount_cumsum(values, discount_array)
    sz = size(values)[1]
    res = similar(values)
    for i = 1:sz
        res[i] = sum(view(values, i:sz) .* view(discount_array, 1:sz-i+1))
    end
    return res
end

function act_loss(actor, log_std, states, actions, adv_est, log_prob_old, hp)
    
    # get actor action predictions from the states
    μ = actor(states)

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

    return p_loss - entropy

end

function crt_loss(critic, states, rewards2go, values_old, hp)
    
    # get critic value predictions from the states
    values = critic(states)

    # calculate clipped value loss
    v_clip = values_old .+ clamp.(values .- values_old, -hp.ϵ, hp.ϵ)
    v_loss1 = (values .- rewards2go) .^ 2
    v_loss2 = (v_clip .- rewards2go) .^ 2
    v_loss = mean(max.(v_loss1, v_loss2))

    return v_loss

end

function setup_optimisers(lr)
    act_opt = ADAM(lr)
    crt_opt = ADAM(lr)
    return act_opt, crt_opt
end

function setup_model(lr)
    
    actor = Chain(
        Dense(state_size, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, action_size)
    ) |> gpu
    critic = Chain(
        Dense(state_size, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 256, relu),
        Dense(256, 1)
    ) |> gpu

    # create the log standard deviations for the model
    log_std = Array{Float32}(undef, action_size)
    for i = eachindex(log_std)
        log_std[i] = -0.5
    end
       
    # setup the optimisers
    act_opt, crt_opt = setup_optimisers(lr)

    return actor, critic, log_std, act_opt, crt_opt
end

function update_ac(actor, critic, log_std, act_opt, crt_opt, states, actions, adv, log_probs, r2g, values, hp)

    # update the actor
    ps = Flux.params(actor, log_std)
    for j = 1:hp.n_updates
        g = gradient(() -> act_loss(actor, log_std, states, actions, adv, log_probs, hp), ps)
        Flux.update!(act_opt, ps, g)
    end

    # update the critic
    ps = Flux.params(critic)
    for j = 1:hp.n_updates
        g = gradient(() -> crt_loss(critic, states, r2g, values, hp), ps)
        Flux.update!(crt_opt, ps, g)
    end

    return
end

function get_batch(actor, critic, log_std, gamma_arr, gamma_lam_arr, hp)
    # clamp log std
    log_std = clamp.(log_std, -20f0, -0.5f0)

    # set the number of environments to run in parallel
    n_parallel = 100

    # setup data buffers for the batch
    batch_size = hp.batch_size
    states_buf = Array{Float32}(undef, state_size, batch_size)
    actions_buf = Array{Float32}(undef, action_size, batch_size)
    log_probs_buf = Array{Float32}(undef, batch_size)
    r2g_buf = Array{Float32}(undef, batch_size)
    adv_buf = Array{Float32}(undef, batch_size)
    val_buf = Array{Float32}(undef, batch_size)

    # setup data buffers for the individual episodes
    states = Array{Float32}(undef, state_size, max_steps, n_parallel)
    actions = Array{Float32}(undef, action_size, max_steps, n_parallel)
    log_probs = Array{Float32}(undef, max_steps, n_parallel)
    values = Array{Float32}(undef, max_steps+1, n_parallel)
    rewards = Array{Float32}(undef, max_steps, n_parallel)
    idxs = ones(Int, n_parallel)

    # setup array for the indivual states
    state = Array{Float32}(undef, state_size, n_parallel)
    
    # reset the states
    for i in 1:n_parallel
        env_reset(view(state, :, i))
    end

    n = 0
    while n < batch_size

        # normalise the states and add to the episode buffer of states
        for i in 1:n_parallel
            states[:, idxs[i], i] = state[:, i]
        end

        # get the actions, log_prob and value estimates
        gpu_state = state |> gpu
        μ = actor(gpu_state)
        μ = μ |> cpu
        value = critic(gpu_state)
        value = value |> cpu
        σ = exp.(log_std)  
        v = σ.^2
        d = Normal.(μ, σ)
        action = rand.(d)
        log_prob = view(sum(-((action .- μ).^2) ./ (2 .* v) .- log_std .- 0.9189385f0, dims=1), 1, :)

        # add to the episode buffers
        for i in 1:n_parallel
            actions[:, idxs[i], i] = action[:, i]
            values[idxs[i], i] = value[i]
            log_probs[idxs[i], i] = log_prob[i]
        end

        # update the states
        for i in 1:n_parallel
            reward, done = env_step(view(state, :, i), view(action, :, i))

            # add the reward to the episode buffer
            rewards[idxs[i], i] = reward

            # check for the episode ending
            if done || idxs[i] == max_steps
                # add a final 0 value
                sz = idxs[i] + 1
                values[sz, i] = 0

                # calculate the rewards to go
                r2g = discount_cumsum(view(rewards, 1:sz-1, i), gamma_arr)

                # calculate the advantage estimates
                δ = view(rewards, 1:sz-1, i) + hp.γ * view(values, 2:sz, i) - view(values, 1:sz-1, i)
                adv_est = discount_cumsum(δ, gamma_lam_arr)

                # update the buffers
                s = n + 1
                sz = min(sz-1, batch_size - n)
                n += sz
                states_buf[:, s:n] = view(states, :, 1:sz, i)
                actions_buf[:, s:n] = view(actions, :, 1:sz, i)
                log_probs_buf[s:n] = view(log_probs, 1:sz, i)
                r2g_buf[s:n] = view(r2g, 1:sz)
                adv_buf[s:n] = view(adv_est, 1:sz)
                val_buf[s:n] = view(values, 1:sz, i)

                # reset this state
                idxs[i] = 0
                env_reset(view(state, :, i))
            end

            # increment the env step index
            idxs[i] += 1
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

    return gpu_log_std, gpu_states_buf, gpu_actions_buf, gpu_adv_buf, gpu_log_probs_buf, gpu_r2g_buf, gpu_val_buf
end

function train(hp)

    # setup arrays to speed up calculation of advantage estimates
    gamma_arr = Array{Float32}(undef, max_steps)
    gamma_lam_arr = Array{Float32}(undef, max_steps)
    for i = eachindex(gamma_arr)
        gamma_arr[i] = hp.γ^(i-1)
        gamma_lam_arr[i] = (hp.γ * hp.λ)^(i-1)
    end

    # setup models and optimisers
    actor, critic, log_std, act_opt, crt_opt = setup_model(hp.lr)

    while true

        # get a batch of data to train on
        st = now()
        log_std, states, actions, adv, log_probs, r2g, values = get_batch(actor, critic, log_std |> cpu, gamma_arr, gamma_lam_arr, hp)
        bt = now() - st

        # update the model critic networks
        st = now()
        update_ac(actor, critic, log_std, act_opt, crt_opt, states, actions, adv, log_probs, r2g, values, hp)
        ut = now() - st

        # timing for performance
        println("batch time: $bt, update time: $ut")
            
    end
end

struct hyper_parameters
    γ::Float32
    λ::Float32
    ϵ::Float32
    c::Float32
    lr::Float32
    ent_coef::Float32
    d::Float32
    n_updates::Int32
    batch_size::Int32
end

function run()

    # set the hyper parameters for the training
    hp = hyper_parameters(0.99, 0.95, 0.2, log(sqrt(2 * π)), 1e-4, 0.01, 0.5 * log(2 * π * ℯ), 1, 5000)

    # run the training
    train(hp)
end

end