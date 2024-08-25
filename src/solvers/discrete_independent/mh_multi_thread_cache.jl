function bpso_thread_cache(problem::DiscreteProblem{T}, args::BPSOArgs)::DiscreteSolution where {T}
    return bpso_thread_cache(Xoshiro(), problem, args)
end

function bpso_thread_cache(
    rng::AbstractRNG,
    problem::DiscreteProblem{T},
    args::BPSOArgs,
)::DiscreteSolution where {T}
    sigmoid(z::Real) = one(z) / (one(z) + exp(-z))

    n_particles = args.n_particles
    n_cycles = args.n_cycles
    c_particle = args.c_particle
    c_global = args.c_global
    w = args.w
    v_max = args.v_max


    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    # init particles and global best
    particle_positions = Vector{Vector{Bool}}()
    for _ = 1:n_particles
        push!(particle_positions, bitrand(rng, length(resources)))
    end
    particle_velocities = Vector{Vector{Float64}}()
    for _ = 1:n_particles
        push!(particle_velocities, zeros(Float64, length(resources)))
    end
    particle_best_positions = deepcopy(particle_positions)
    particle_best_costs = [
        sro_target_function(resources[x], p_target, v_target) for
        x in particle_best_positions
    ]

    # initialize global best to full selection
    global_best_position = ones(Bool, length(resources))
    global_best_cost = sro_target_function(resources, p_target, v_target)

    known_combinations = Vector{Dict{Set{Int64},T}}()
    for _ in 1:Threads.nthreads()
        push!(known_combinations, Dict{Set{Int64},T}())
    end

    # velocity update: v(t+1) = w * v(t) + c_particle R1 (local_best - position) + c_global R2 (g_best - position)
    # have to apply this bitwise
    #
    # x(t+1) = 0 if rand() >= S(v(t+1))
    #          1 else
    #
    # local and global best are set as one may expect
    for _ = 1:n_cycles

        Threads.@threads for i = 1:n_particles
            # update velocity
            r1 = rand(rng)
            r2 = rand(rng)

            v = particle_velocities[i]
            local_best = particle_best_positions[i]
            local_eval = particle_best_costs[i]
            pos = particle_positions[i]

            v_new =
                w * v +
                c_particle * r1 * (local_best - pos) +
                c_global * r2 * (global_best_position - pos)
            particle_velocities[i] = [min(v_max, v) for v in v_new]


            # update position
            bitflip = rand(rng, length(resources))
            pos_new =
                Bool[bitflip[i] >= sigmoid(v_new[i]) ? 0 : 1 for i in eachindex(bitflip)]
            particle_positions[i] = pos_new

            # evaluate
            eval_new = Inf
            r_set = Set(pos_new)
            found = false
            for t in 1:Threads.nthreads()
                if r_set in keys(known_combinations[t])
                    eval_new = known_combinations[t][r_set]
                    found = true
                    break
                end
            end

            if !found
                eval_new = sro_target_function(resources[pos_new], p_target, v_target)
                known_combinations[Threads.threadid()][r_set] = eval_new
            end

            if eval_new < local_eval
                particle_best_costs[i] = eval_new
                particle_best_positions[i] = pos_new
            end
        end

        c_min, c_min_index = findmin(particle_best_costs)
        if c_min < global_best_cost
            global_best_cost = c_min
            global_best_position = particle_best_positions[c_min_index]
        end
    end

    return DiscreteSolution(
        problem,
        collect(1:length(resources))[global_best_position],
        resources[global_best_position],
        global_best_cost,
    )
end

function n_thread_evo_thread_cache(problem::DiscreteProblem, n_steps::Int64)::DiscreteSolution
    return n_thread_evo_thread_cache(Xoshiro(), problem, n_steps)
end

function n_thread_evo_thread_cache(
    rng::AbstractRNG,
    problem::DiscreteProblem{T},
    n_steps::Int64,
)::DiscreteSolution where {T}
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    best_cost = sro_target_function(resources, p_target, v_target)
    select_vector = ones(Bool, length(resources))
    p_bit_flip = 1.0 / length(resources)

    thread_bests = [best_cost for _ in 1:Threads.nthreads()]
    thread_vectors = [select_vector for _ in 1:Threads.nthreads()]

    known_combinations = Vector{Vector{Vector{Bool}}}()
    for _ in 1:Threads.nthreads()
        push!(known_combinations, Vector{Vector{Bool}}())
    end

    for _ = 1:n_steps
        @sync Threads.@threads for i = 1:Threads.nthreads()
            new_select_vector = zeros(Bool, length(resources))
            for i in eachindex(new_select_vector)
                if rand(rng) <= p_bit_flip
                    new_select_vector[i] = !select_vector[i]
                else
                    new_select_vector[i] = select_vector[i]
                end
            end

            found = false
            for i in 1:Threads.nthreads()
                if new_select_vector in known_combinations[i]
                    # no way to immprove, skip
                    found = true
                    break
                end
            end

            if found
                break
            end

            new_cost = sro_target_function(resources[new_select_vector], p_target, v_target)
            push!(known_combinations[Threads.threadid()], new_select_vector)

            if new_cost < best_cost
                thread_bests[Threads.threadid()] = new_cost
                thread_vectors[Threads.threadid()] = new_select_vector
            end
        end

        m, idx = findmin(thread_bests)
        if m < best_cost
            best_cost = m
            select_vector = thread_vectors[idx]
        end
    end

    return DiscreteSolution(
        problem,
        collect(1:length(resources))[select_vector],
        resources[select_vector],
        best_cost,
    )
end
