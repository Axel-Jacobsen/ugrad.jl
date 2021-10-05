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
  µ.zero_grad(c)
  d = µ.Value(5)
  c *= 5 + d
  @test c.data == 70
  @test c.grad == 0
  e = 5 + d
  @test e.data == 10
end


@testset "Basic Network Tests" begin
  l1 = µ.Layer(5, 10)
  l2 = µ.Layer(10, 1)

  for v in µ.parameters(l1)
    @test v.grad == 0.
  end

  for v in µ.parameters(l2)
    @test v.grad == 0.
  end

  x = [1,2,3,4,5]
  out1 = µ.forward!(l1, x)
  out2 = µ.forward!(l2, out1)
  µ.backward!(out2[1])

  µ.zero_grad(l1)
  µ.zero_grad(l2)

  for v in µ.parameters(l1)
    @test v.grad == 0.
  end

  for v in µ.parameters(l2)
    @test v.grad == 0.
  end
end
