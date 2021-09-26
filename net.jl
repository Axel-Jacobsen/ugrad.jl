module µNet

include("µengine.jl")
using µEngine: Value, backward
export Value, backward

include("µmodule.jl")
using µModule: Module, Neuron, Layer, zero_grads, parameters, forward
export Module, Neuron, Layer, zero_grads, parameters, forward

end
