using Mango
import Mango.handle_message

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
    termination_timeout_s::Float64
)::PropagatingAgent

    agent = PropagatingAgent(
        resource,
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

    register(container, agent)
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
    agent_set::Set{AgentAddress}
    cost::T
end

function handle_message(agent::PropagatingAgent, content::Any, meta::AbstractDict)
    # technically not threadsafe but should not matter here?
    agent.t_last_message = time()

    if !haskey(meta, TYPE_FIELD)
        return
    end

    if meta[TYPE_FIELD] == RESOURCE_MESSAGE_TYPE
        schedule(agent, InstantTaskData()) do
            handle_resource_message(agent, content)
        end
    end

    if meta[TYPE_FIELD] == LOCAL_BEST_MESSAGE_TYPE
        # TODO implement me
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
    new_cost =
        sro_target_function(agent.aggregate_resource, agent.p_target, agent.v_target)


    # schedule message propagation to all neighbors
    for neighbor in agent.neighbors
        schedule(agent, InstantTaskData()) do
            send_message(agent, msg, neighbor)
        end
    end
end

function run(agent::PropagatingAgent)
    # TODO send first info message out

    # agent is in information gathering loop
    agent.t_last_message = time()
    while true
        if time() - agent.t_last_message > agent.information_timeout_s
            break
        end
        yield()
    end

    # TODO send first local best message out

    # agent is in bests solution gathering loop
    agent.t_last_message = time()
    while true
        if time() - agent.t_last_message > agent.termination_timeout_s
            break
        end
        yield()
    end
end