module µNet

include("engine.jl")
using .µEngine: Value, backward!
export Value, backward!

include("module.jl")
using .µModule: Module, Neuron, Layer, zero_grads, parameters, forward!
export Module, Neuron, Layer, zero_grads, parameters, forward!

end
