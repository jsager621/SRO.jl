"""
Container for the various arguments of the BPSO algorithm.
- `n_particle` - number of particles to use
- `n_cycles`   - number of cycles until termination
- `c_particle` - acceleration constant for local best term
- `c_global`   - acceleration constant for global best term
- `w`          - inertial constant
- `v_max`      - maximum velocity of a particle
"""
@kwdef struct BPSOArgs
    n_particles::Int64 = 0
    n_cycles::Int64 = 0
    c_particle::Float64 = 1.43
    c_global::Float64 = 1.43
    w::Float64 = 0.69
    v_max::Float64 = 4.0
end

"""
Binary PSO algorithm as originally described by Kennedy and Eberhart (1997).
This implementation uses two acceleration constants, a global best topology, and
an intertia constant.

For a description of the input parameters see the `BPSOArgs` struct.
"""
function bpso(problem::DiscreteProblem{T}, args::BPSOArgs)::DiscreteSolution where {T}
    return bpso(Xoshiro(), problem, args)
end

function bpso(
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

    known_combinations = ConcurrentDict{Set{Int64},T}()

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
            if r_set in keys(known_combinations)
                eval_new = known_combinations[r_set]
            else
                eval_new = sro_target_function(resources[pos_new], p_target, v_target)
                known_combinations[r_set] = eval_new
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

"""
Container for the various arguments of the ACO algorithm.
- `n_ants`     - number of ants to use
- `n_runs`     - number of runs until termination
- `phero_init` - initial pheromone value on each note
- `phero_add`  - amount of pheromone added to a node on run win
- `decay_rate` - relative rate of pheromone decay (should be 0 < `decay_rate` < 1)
"""
@kwdef struct ACOArgs
    n_ants::Int64 = 10
    n_runs::Int64 = 10
    phero_init::Float64 = 10.0
    phero_add::Float64 = 1.0
    decay_rate::Float64 = 0.9
end

"""
ACO algorithm for  discrete SRO. For each run in n_runs, each ant selects resources
in a random order and determines the best combination of resources in this order.
Ants then apply a pheromone value to the best combination found in the run.
Pheromones determine the probability of selecting a resource on the next run.

In this implementation P(resource) = pheromone(resource)/total_pheromone.
Pheromones are initialized with an equal starting value `phero_init`.
Pheromones decay after each run by the rate given by `decay_rate`.
"""
function aco(problem::DiscreteProblem{T}, args::ACOArgs)::DiscreteSolution where {T}
    return aco(Xoshiro(), problem, args)
end

function aco(
    rng::AbstractRNG,
    problem::DiscreteProblem{T},
    args::ACOArgs,
)::DiscreteSolution where {T}
    n_ants = args.n_ants
    n_runs = args.n_runs
    phero_init = args.phero_init
    phero_add = args.phero_add
    decay_rate = args.decay_rate

    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    pheromones = [phero_init for _ = 1:length(resources)]

    # unlike BPSO, its a vector here because order matters!
    # maps an order of resources to its cutoff (Int64)
    known_combinations = ConcurrentDict{Vector{Int64}, Int64}()

    for _ = 1:n_runs
        # compute probabilities for this run once
        total_pheromone = sum(pheromones)
        probabilities = ProbabilityWeights([x / total_pheromone for x in pheromones])
        resource_best_counters =
            _best_ant_solutions!(rng, probabilities, resources, n_ants, p_target, v_target, known_combinations)

        # decay old pheromone before applying new one
        pheromones = pheromones .* decay_rate

        for i in eachindex(pheromones)
            pheromones[i] += resource_best_counters[i] * phero_add
        end
    end

    cutoff, solution_cost =
        _best_set_in_order(resources, sortperm(pheromones, rev = true), p_target, v_target)
    solution_indices = sortperm(pheromones, rev = true)[1:cutoff]

    return DiscreteSolution(
        problem,
        solution_indices,
        resources[solution_indices],
        solution_cost,
    )
end

"""
Runs `n_ants` on the given resource set with given probabilities for selection.
Returns the number of times each resource was best.
"""
function _best_ant_solutions!(
    rng::AbstractRNG,
    probabilities::ProbabilityWeights,
    resources::Vector{DiscreteResource{T}},
    n_ants::Int64,
    p_target::T,
    v_target::Int64,
    known_combinations::ConcurrentDict{Vector{Int64}, Int64}
)::Vector{Int64} where {T}
    ant_outputs = Vector{Vector{Int64}}()
    for _ = 1:n_ants
        push!(ant_outputs, zeros(Int64, length(resources)))
    end

    Threads.@threads for a_id = 1:n_ants
        ant_path = sample(
            rng,
            collect(1:length(resources)),
            probabilities,
            length(resources),
            replace = false,
        )

        cutoff = -1
        if ant_path in keys(known_combinations)
            cutoff = known_combinations[ant_path]
        else
            cutoff, _ = _best_set_in_order(resources, ant_path, p_target, v_target)
        end
        
        ant_indices = ant_path[1:cutoff]
        for i in ant_indices
            ant_outputs[a_id][i] += 1
        end
    end

    return sum(ant_outputs)
end

"""
Finds the best subset of resources in the given sorting order.
Returns the index of the last resource in the best set and the solution cost.
"""
function _best_set_in_order(
    resources::Vector{DiscreteResource{T}},
    sorting::Vector{Int64},
    p_target::T,
    v_target::Int64,
)::Tuple{Int64,Float64} where {T}
    ordered_resources = resources[sorting]

    intermediate_set = Vector{DiscreteResource{T}}()
    best_cost = Inf
    cutoff = 1

    for (i, r) in enumerate(ordered_resources)
        push!(intermediate_set, r)
        cost = sro_target_function(intermediate_set, p_target, v_target)
        if cost < best_cost
            best_cost = cost
            cutoff = i
        end
    end

    return (cutoff, best_cost)
end

"""
Basic one plus one evolutionary algorithm.
Starts with all resources as the starting solution.
In each step the current solution gets modified by randomly flipping the selection bits.
The probability of each bit to get flipped is 1/length(resources).
After `n_steps` the best solution found so far is returned.
"""
function one_plus_one_evo(problem::DiscreteProblem, n_steps::Int64)::DiscreteSolution
    return one_plus_one_evo(Xoshiro(), problem, n_steps)
end

function one_plus_one_evo(
    rng::AbstractRNG,
    problem::DiscreteProblem,
    n_steps::Int64,
)::DiscreteSolution
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    best_cost = sro_target_function(resources, p_target, v_target)
    select_vector = ones(Bool, length(resources))
    p_bit_flip = 1.0 / length(resources)

    for _ = 1:n_steps
        new_select_vector = zeros(Bool, length(resources))
        for i in eachindex(new_select_vector)
            if rand(rng) <= p_bit_flip
                new_select_vector[i] = !select_vector[i]
            else
                new_select_vector[i] = select_vector[i]
            end
        end

        new_cost = sro_target_function(resources[new_select_vector], p_target, v_target)

        if new_cost < best_cost
            best_cost = new_cost
            select_vector = new_select_vector
        end
    end

    return DiscreteSolution(
        problem,
        collect(1:length(resources))[select_vector],
        resources[select_vector],
        best_cost,
    )
end