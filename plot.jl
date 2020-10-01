using Plots
using CSV
pyplot()

function plot_scores(exp_names)

    # open the csv file that contains the scores and load them
    f = CSV.File("scores.csv")
    sz = size(f)[1]
    filenames = Array{String}(undef,sz)
    scores = Array{Float32}(undef, sz)
    i = 1
    for row in f
        filenames[i] = row.filename
        scores[i] = row.score
        i += 1
    end

    # plot the results
    for i = eachindex(exp_names)
        exp_steps = Array{Int32}(undef, sz)
        exp_scores = Array{Float32}(undef, sz)
        n = 0
        name = exp_names[i]
        for j = eachindex(filenames)
            if match(Regex("$name/"), filenames[j]) !== nothing
                n += 1
                s = match(r"_\d+[.]", filenames[j])
                step = parse(Int, match(r"\d+", s.match).match)
                exp_steps[n] = step
                exp_scores[n] = scores[j]
            end
        end
        if n > 0
            exp_steps = view(exp_steps, 1:n)
            exp_scores = view(exp_scores, 1:n)
            if i == 1
                display(scatter(exp_steps, exp_scores, label=name, markersize=7, markerstrokewidth=0.01))
            else
                display(scatter!(exp_steps, exp_scores, label=name, markersize=7, markerstrokewidth=0.01))
            end
        end
    end

    # pause until seen the plots and pressed a key
    k = readline()

end

exp_names = ["tws-orig", "tws-compare", "tws-compare-5models", "jl-multi", "jl-tanh"]
plot_scores(exp_names)
