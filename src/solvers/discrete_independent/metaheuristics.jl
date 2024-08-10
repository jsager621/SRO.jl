"""
Container for the various arguments of the BPSO algorithm.
`n_particle` - number of particles to use
`n_cycles`   - number of cycles until termination
`c_particle` - acceleration constant for local best term
`c_global`   - acceleration constant for global best term
`w`          - inertial constant
`v_max`      - maximum velocity of a particle
"""
@kwdef struct BPSOArgs
    n_particles::Int64 = 0
    n_cycles::Int64 = 0
    c_particle::Float64 = 1.43
    c_global::Float64 = 1.43
    w::Float64 = 0.69
    v_max::Float64 = 4.0
end

function bpso(problem::DiscreteProblem, args::BPSOArgs)::DiscreteSolution
    return bpso(Xoshiro(), problem, args)
end

function bpso(rng::AbstractRNG, problem::DiscreteProblem, args::BPSOArgs)::DiscreteSolution
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
    global_best_positions = ones(Bool, length(resources))
    global_best_costs = sro_target_function(resources, p_target, v_target)

    # velocity update: v(t+1) = w * v(t) + c_particle R1 (local_best - position) + c_global R2 (g_best - position)
    # have to apply this bitwise
    #
    # x(t+1) = 0 if rand() >= S(v(t+1))
    #          1 else
    #
    # local and global best are set as one may expect
    for _ = 1:n_cycles
        intermediate_g_best = global_best_positions
        intermediate_best_costs = global_best_costs

        for i = 1:n_particles
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
                c_global * r2 * (global_best_positions - pos)
            particle_velocities[i] = [min(v_max, v) for v in v_new]


            # update position
            bitflip = rand(rng, length(resources))
            pos_new =
                Bool[bitflip[i] >= sigmoid(v_new[i]) ? 0 : 1 for i in eachindex(bitflip)]
            particle_positions[i] = pos_new

            # evaluate
            eval_new = sro_target_function(resources[pos_new], p_target, v_target)

            if eval_new < local_eval
                particle_best_costs[i] = eval_new
                particle_best_positions[i] = pos_new
            end

            if eval_new < intermediate_best_costs
                intermediate_best_costs = eval_new
                intermediate_g_best = pos_new
            end
        end

        global_best_positions = intermediate_g_best
        global_best_costs = intermediate_best_costs
    end

    return DiscreteSolution(
        problem,
        collect(1:length(resources))[global_best_positions],
        resources[global_best_positions],
        global_best_costs,
    )
end


function aco(problem::DiscreteProblem)::DiscreteSolution

end