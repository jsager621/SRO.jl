using SRO
using Plots

THIS_DIR = @__DIR__

function plot_results(results::Dict{String, Dict{Int64, Float64}}, n_resources::Int64)
    p = plot(legend=:bottomright)

    for algo in keys(results)
        data_points = zeros(Float64, n_resources)
        for k in keys(results[algo])
            data_points[k] = results[algo][k]
        end

        plot!(p, data_points, label=algo, markershape=:auto, yaxis=:log)
    end

    outfile = THIS_DIR * "/results.png"
    savefig(p, outfile)
end

function main()
    N_SAMPLES = 50
    N_RESOURCES_MAX = 30

    rv = [0.1, 0.4, 0.5]
    cost = [0.0, 1.0, 2.0]
    r = DiscreteResource(rv, cost)

    # algo > n_resources > mean benchmark time
    results = Dict{String, Dict{Int64, Float64}}()

    # bpso params
    args = BPSOArgs(; n_particles = 10, n_cycles = 2)
    # evo params
    n_steps_threaded = 50
    n_steps_single = n_steps_threaded * Threads.nthreads()

    p_target = 0.0
    v_target = 0

    bpso_algos = [bpso_no_cache, bpso_simple_cache, bpso_thread_cache]
    evo_algos = [n_thread_evo_no_cache, n_thread_evo_simple_cache, n_thread_evo_thread_cache]
    comp = [one_plus_one_evo]

    #---------------------------------------
    # dummy run everything to remove compile times
    #---------------------------------------
    p1 = DiscreteProblem([r], p_target, v_target)
    for a in bpso_algos
        a(p1, args)
    end
    for a in evo_algos
        a(p1, n_steps_threaded)
    end
    for a in comp
        a(p1, n_steps_single)
    end

    #---------------------------------------
    # actual runs
    #---------------------------------------

    for a in vcat(bpso_algos, evo_algos, comp)
        results[string(a)] = Dict{Int64, Float64}()
    end

    for n_resources in 1:N_RESOURCES_MAX
        problem = DiscreteProblem([r for _ in 1:n_resources], p_target, v_target)

        for a in bpso_algos
            times = Vector{Float64}()
            for _ in 1:N_SAMPLES
                t_start = time()
                a(problem, args)
                t_end = time()
                push!(times, t_end - t_start)
            end
            results[string(a)][n_resources] = mean(times)
        end

        for a in evo_algos
            times = Vector{Float64}()
            for _ in 1:N_SAMPLES
                t_start = time()
                a(problem, n_steps_threaded)
                t_end = time()
                push!(times, t_end - t_start)
            end
            results[string(a)][n_resources] = mean(times)
        end

        for a in comp
            times = Vector{Float64}()
            for _ in 1:N_SAMPLES
                t_start = time()
                a(problem, n_steps_single)
                t_end = time()
                push!(times, t_end - t_start)
            end
            results[string(a)][n_resources] = mean(times)
        end
    end

    plot_results(results, N_RESOURCES_MAX)    
end


main()