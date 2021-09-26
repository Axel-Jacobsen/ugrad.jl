#! /usr/local/bin/julia

include("µnet.jl")
include("engine.jl")

import .µNet as µ
import .Engine as E
using Test


verbose = true

@testset "Basic Engine Tests" begin
  a = E.Value(3)
  b = E.Value(4)
  c = a + b
  @test c.data == a.data + b.data
  E.backward(c)
  @test a.grad == 1
  @test b.grad == 1
  @test c.grad == 1
  d = E.Value(5)
  c *= 5
end


@testset "Basic Network Tests" begin
  c = µ.Neuron(5)
  out = µ.forward(c, [1,2,3,4,5])
  println(out)
  E.backward(out)
  println(out)
end
