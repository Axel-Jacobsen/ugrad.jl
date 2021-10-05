module µModule
export Module, Neuron, Layer, zero_grad, parameters, forward!


import Main.µNet as µ
import Distributions as d


abstract type Module end


function parameters(m::Module)::Vector{µ.Value}
  Vector{µ.Value}()
end


function zero_grad(v::µ.Value)
  v.grad = 0
end


function zero_grad(m::Module)
  for v in parameters(m)
    zero_grad(v)
  end
end


struct Neuron <: Module
  weights::Vector{µ.Value}
  bias::µ.Value
  function Neuron(nin::Int)
    weights = [µ.Value(rand(d.Uniform(-1, 1))) for _ in 1:nin]
    bias = µ.Value(rand(d.Uniform(-1, 1)))
    new(weights, bias)
  end
end

function parameters(n::Neuron)::Vector{µ.Value}
  [n.weights; n.bias]
end

function forward!(n::Neuron, v::Vector{T} where T = Union{Real, µ.Value})
  mapreduce(((p,v),) -> p*v, +, zip(parameters(n), v))
end

Base.show(io::IO, x::Neuron) = print(io, "Neuron(nin=$(length(x.weights)))")
Base.show(io::IO, m::MIME"text/plain", x::Neuron) = print(io, "Neuron(nin=$(length(x.weights)))")


struct Layer <: Module
  neurons::Vector{Neuron}
  function Layer(nin::Int, nout::Int)
    neurons = [Neuron(nin) for _ in 1:nout]
    new(neurons)
  end
end

function parameters(l::Layer)::Vector{µ.Value}
  reduce(vcat, [parameters(n) for n in l.neurons])
end

function forward!(l::Layer, v::Vector{T} where T = Union{Real, µ.Value})
  [forward!(n, v) for n in l.neurons]
end

Base.show(io::IO, x::Layer) = print(io, "Layer(nin=$(length(x.neurons[1].weights)), nout=$(length(x.neurons)))")
Base.show(io::IO, m::MIME"text/plain", x::Layer) = print(io, "Layer(nin=$(length(x.neurons[1].weights)), nout=$(length(x.neurons)))")

end
