using Mango
import Mango.handle_message
using Sockets: InetAddr, @ip_str

#---------------------------------------
# Simple Topology-Dependent Heuristic
#---------------------------------------
# """
# Agent that samples the SRO solution space in a neighborhood topology in these steps:
# * Send information on its own resource to its `neighbors`.
# * When new resource information arrives: 
#   * append the resource to the local vector
#   * evaluate the local vector and save the result
#   * send the resource information to other `neighbors`

# * When no new information has arrived in `information_timeout_s` seconds:
#   * send the best known combination to `neighbors`.
# """
# TODO add parametric type once this is fixed in Mango.jl

@agent struct PropagatingAgent
    base_resource::DiscreteResource{Float64}
    aggregate_resource::DiscreteResource{Float64}
    p_target::Float64
    v_target::Int64
    agent_order::Vector{AgentAddress}
    neighbors::Vector{AgentAddress}
    information_timeout_s::Float64
    termination_timeout_s::Float64
    best_agents::Vector{AgentAddress}
    best_cost::Float64
    t_last_message::Float64
end

"""
Convenience method to create and register a PropagatinAgent to the
given `container` with the local `resource`, `target`, and timeouts.
"""
function propagating_agent_factory(
    container::Container,
    resource::DiscreteResource{Float64},
    p_target::Float64,
    v_target::Int64,
    information_timeout_s::Float64,
    termination_timeout_s::Float64,
)::PropagatingAgent

    agent = PropagatingAgent(
        resource,
        deepcopy(resource),
        p_target,
        v_target,
        Vector{AgentAddress}(),
        Vector{AgentAddress}(),
        information_timeout_s,
        termination_timeout_s,
        Vector{AgentAddress}(),
        Inf,
        time(),
    )

    Mango.register(container, agent)
    push!(agent.agent_order, address(agent))

    agent.best_agents = AgentAddress[address(agent)]
    agent.best_cost = sro_target_function([resource], p_target, v_target)

    return agent
end

TYPE_FIELD = "type"
RESOURCE_MESSAGE_TYPE = "resource_message"
struct ResourceMessage{T<:AbstractFloat}
    source::AgentAddress
    p::Vector{T}
    c::Vector{T}
end

LOCAL_BEST_MESSAGE_TYPE = "local_best"
struct LocalBestMessage{T<:AbstractFloat}
    agent_set::Vector{AgentAddress}
    cost::T
end

function handle_message(agent::PropagatingAgent, content::Any, meta::AbstractDict)
    agent.t_last_message = time()

    if !haskey(meta, TYPE_FIELD)
        return
    end

    if meta[TYPE_FIELD] == RESOURCE_MESSAGE_TYPE
        handle_resource_message(agent, content)
    end

    if meta[TYPE_FIELD] == LOCAL_BEST_MESSAGE_TYPE
        handle_local_best_message(agent, content)
    end
end

function handle_local_best_message(agent::PropagatingAgent, msg::LocalBestMessage)
    # only propagate if its better than what we already know
    # (because we already sent that one out then)
    if msg.cost < agent.best_cost
        agent.best_cost = msg.cost
        agent.best_agents = deepcopy(msg.agent_set)
        send_local_best_info(agent)
    end
end

function handle_resource_message(agent::PropagatingAgent, msg::ResourceMessage)
    # if we already got this information, do nothing
    if msg.source in agent.agent_order
        return
    end

    # update local stuff with new info
    push!(agent.agent_order, msg.source)
    new_res = DiscreteResource(msg.p, msg.c)
    agent.aggregate_resource = combine(agent.aggregate_resource, new_res)
    new_cost = sro_target_function([agent.aggregate_resource], agent.p_target, agent.v_target)

    if new_cost < agent.best_cost
        agent.best_cost = new_cost
        agent.best_agents = deepcopy(agent.agent_order)
    end

    # schedule message propagation to all neighbors
    for neighbor in agent.neighbors
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=RESOURCE_MESSAGE_TYPE)
        end
    end
end

function send_resource_info(agent::PropagatingAgent)
    msg = ResourceMessage(address(agent), agent.base_resource.p, agent.base_resource.c)

    for neighbor in agent.neighbors
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=RESOURCE_MESSAGE_TYPE)
        end
    end
end

function send_local_best_info(agent::PropagatingAgent)
    msg = LocalBestMessage(agent.best_agents, agent.best_cost)

    for neighbor in agent.neighbors
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=LOCAL_BEST_MESSAGE_TYPE)
        end
    end
end

function run_agent(agent::PropagatingAgent)
    # start information propagation
    send_resource_info(agent)

    # agent is in information gathering loop
    agent.t_last_message = time()
    while true
        if time() - agent.t_last_message > agent.information_timeout_s
            break
        end
        yield()
    end

    # start local best propagation
    send_local_best_info(agent)

    # agent is in bests solution gathering loop
    agent.t_last_message = time()
    while true
        if time() - agent.t_last_message > agent.termination_timeout_s
            break
        end
        yield()
    end
end

"""
Run the discrete `problem` with the PropagatingAgent algorithm 
in a single container.
One agent is created for each problem resource.
Agent neighbourhoods are given by the `adjacency_matrix`.
Agents may not be their own neighbor and entries on the diagonal
are ignored.
"""
function run_propagated_agent_problem(
    problem::DiscreteProblem,
    adjacency_matrix::Matrix{Bool},
    container_addr::InetAddr,
    information_timeout::Float64,
    termination_timeout::Float64,
)::DiscreteSolution
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    if !(size(adjacency_matrix)[1] == size(adjacency_matrix)[2])
        throw(ArgumentError("Adjacency matrix must be square!"))
    end

    if !(size(adjacency_matrix)[1] == length(resources))
        throw(
            ArgumentError(
                "Dimension mismatch between number of resources and adjacency matrix!",
            ),
        )
    end

    c = Container()
    c.protocol = TCPProtocol(address = container_addr)

    agents = Vector{PropagatingAgent}()

    for r in resources
        new_agent = propagating_agent_factory(
            c,
            r,
            p_target,
            v_target,
            information_timeout,
            termination_timeout,
        )
        push!(agents, new_agent)
    end
    
    # set neighborhoods
    agent_addresses = [address(a) for a in agents]
    
    for i in 1:length(resources)
        for j in 1:length(resources)
            if i == j
                continue
            end

            # i has j as neighbor
            if adjacency_matrix[i,j]
                push!(agents[i].neighbors, agent_addresses[j])
            end
        end
    end

    # run container
    wait(Threads.@spawn start(c))

    # run agents
    @sync for a in agents
        Threads.@spawn run_agent(a)
    end

    wait(Threads.@spawn shutdown(c))

    # read terminated info and make output
    best_agents = agents[1].best_agents # agent adresses
    best_cost = agents[1].best_cost
    output_indices = Vector{Int64}()

    for b in best_agents
        # agent index matches resource index
        index = findfirst(x->x==b, agent_addresses)
        push!(output_indices, index)
    end

    return DiscreteSolution(
        problem,
        output_indices,
        resources[output_indices],
        best_cost
    )
end
