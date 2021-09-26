#! /usr/local/bin/julia


include("µnet.jl")
import .µNet as µ

using Test


verbose = true

@testset "Basic Engine Tests" begin
  a = µ.Value(3)
  b = µ.Value(4)
  c = a + b
  @test c.data == a.data + b.data
  µ.backward(c)
  @test a.grad == 1
  @test b.grad == 1
  @test c.grad == 1
  d = µ.Value(5)
  c *= 5
end


@testset "Basic Network Tests" begin
  c = µ.Neuron(5)
  out = µ.forward(c, [1,2,3,4,5])
  println(out)
  µ.backward(out)
  println(out)
end
