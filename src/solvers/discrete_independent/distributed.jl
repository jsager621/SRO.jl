using Mango
using Graphs

@agent struct PropagatingAgent
    base_resource::DiscreteResource{Float64}
    aggregate_resource::DiscreteResource{Float64}
    n_resources::Int64
    p_target::Float64
    v_target::Int64
    agent_order::Vector{AgentAddress}
    best_agents::Vector{AgentAddress}
    best_cost::Float64
    best_msg_sources::Vector{AgentAddress}
end

function propagating_agent_factory(
    resource::DiscreteResource{Float64},
    n_resources::Int64,
    p_target::Float64,
    v_target::Int64,
)::PropagatingAgent
    agent = PropagatingAgent(
        resource,
        deepcopy(resource),
        n_resources,
        p_target,
        v_target,
        Vector{AgentAddress}(),
        Vector{AgentAddress}(),
        Inf,
        Vector{AgentAddress}()
    )

    agent.best_agents = AgentAddress[]
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
    source::AgentAddress
    agent_set::Vector{AgentAddress}
    cost::T
end

function Mango.handle_message(agent::PropagatingAgent, content::Any, meta::AbstractDict)
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
    # if we already got this information, do nothing
    if msg.source in agent.best_msg_sources
        return
    end

    push!(agent.best_msg_sources, msg.source)

    if msg.cost < agent.best_cost
        agent.best_cost = msg.cost
        agent.best_agents = deepcopy(msg.agent_set)
    end

    # schedule message propagation to all neighbors
    for neighbor in topology_neighbors(agent)
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=LOCAL_BEST_MESSAGE_TYPE)
        end
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
    for neighbor in topology_neighbors(agent)
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=RESOURCE_MESSAGE_TYPE)
        end
    end
end

function send_resource_info(agent::PropagatingAgent)
    msg = ResourceMessage(address(agent), agent.base_resource.p, agent.base_resource.c)

    for neighbor in topology_neighbors(agent)
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=RESOURCE_MESSAGE_TYPE)
        end
    end
end

function send_local_best_info(agent::PropagatingAgent)
    msg = LocalBestMessage(address(agent), agent.best_agents, agent.best_cost)

    for neighbor in topology_neighbors(agent)
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor; type=LOCAL_BEST_MESSAGE_TYPE)
        end
    end
end

function run_agent(agent::PropagatingAgent)
    # should be registered now, add own addres to list
    push!(agent.agent_order, address(agent))
    push!(agent.best_msg_sources, address(agent))
    
    # start information propagation
    send_resource_info(agent)

    # agent is in information gathering loop
    while length(agent.agent_order) < agent.n_resources
        yield()
    end

    # start local best propagation
    send_local_best_info(agent)

    # agent is in bests solution gathering loop
    while length(agent.best_msg_sources) < agent.n_resources
        yield()
    end
end


"""
Run the discrete `problem` with the PropagatingAgent algorithm 
in a single container.
One agent is created for each problem resource.
Agent neighbourhoods are given by the `topology`.
Agents may not be their own neighbor and entries on the diagonal
are ignored.
"""
function propagated_agent_solver(
    problem::DiscreteProblem,
    topology::Topology,
    host::String,
    port::Int64
)::DiscreteSolution
    resources = problem.resources
    p_target = problem.p_target
    v_target = problem.v_target

    if !(nv(topology.graph) == length(resources))
        throw(
            ArgumentError(
                "Dimension mismatch between number of resources and topology!",
            ),
        )
    end

    c = create_tcp_container(host, port)
    agents = Vector{PropagatingAgent}()

    # make agents
    for r in resources
        new_agent = propagating_agent_factory(
            r,
            length(resources),
            p_target,
            v_target
        )
        Mango.register(c, new_agent)
        push!(agents, new_agent)
    end
    agent_addresses = [address(a) for a in agents]

    # set topology
    i = 1
    per_node(topology) do node
        add!(node, agents[i])
        i += 1
    end

    activate(c) do
        # run agents
        @sync for a in agents
            Threads.@spawn run_agent(a)
        end
    end

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
