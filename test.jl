#! /usr/local/bin/julia


include("net.jl")
import .µNet as µ

using Test


verbose = true


@testset "Basic Engine Tests" begin
  a = µ.Value(3)
  b = µ.Value(4)
  c = a + b
  @test c.data == a.data + b.data
  µ.backward!(c)
  @test a.grad == 1
  @test b.grad == 1
  @test c.grad == 1
  d = µ.Value(5)
  c *= 5
end


@testset "Basic Network Tests" begin
  l1 = µ.Layer(5, 10)
  l2 = µ.Layer(10, 1)
  out1 = µ.forward!(l1, [1,2,3,4,5])
  out2 = µ.forward!(l2, out1)[1]
  println(out2)
  µ.backward!(out2)
  println(out1)
  println(out2)
end
