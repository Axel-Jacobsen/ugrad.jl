module µNet

module Engine
export Value, backward


mutable struct Value
  data::Float64
  grad::Float64
  _backward::Function
  _prev::Set{Value}
  _op::String
  Value(data; _children::Set=Set(), _op::String="") = new(data, 0, ()->nothing, _children, _op)
end


function Base.:+(x::Union{Number, Value}, y::Union{Number, Value})::Value
  """ x + y = z
  """
  x = isa(x, Value) ? x : Value(x)
  y = isa(y, Value) ? y : Value(y)
  z = Value(x.data + y.data, _children=Set([x, y]), _op="+")

  function _backward()
    x.grad += z.grad
    y.grad += z.grad
  end
  z._backward = _backward

  z
end


function Base.:*(x::Union{Number, Value}, y::Union{Number, Value})::Value
  """ x * y = z
  """
  x = isa(x, Value) ? x : Value(x)
  y = isa(y, Value) ? y : Value(y)
  z = Value(x.data * y.data, _children=Set([x, y]), _op="*")

  function _backward()
    x.grad += y.data * z.grad
    y.grad += x.data * z.grad
  end
  z._backward = _backward

  z
end


function Base.:^(x::Union{Number, Value}, y::Number)::Value
  """ x ^ y = z
  y is Number, not Value. I tried with y being a value too, but you
  pretty quickly run into cases where x < 0, and then the derivative
  of the exponent (= log(x) x^y * z.grad) becomes complex.
  """
  x = isa(x, Value) ? x : Value(x)
  z = Value(x.data^y, _children=Set([x]), _op="^")

  function _backward()
    x.grad += y * x.data^(y-1) * z.grad
  end
  z._backward = _backward

  z
end


function Base.:-(x::Union{Number, Value}, y::Union{Number, Value})::Value
  """ x - y = z
  """
  x = isa(x, Value) ? x : Value(x)
  y = isa(y, Value) ? y : Value(y)
  x + (-y)
end


function Base.:inv(x::Value)::Value
  """ x^-1 = z
  """
  x = isa(x, Value) ? x : Value(x)
  z = Value(1 / x.data, _children=Set([x]), _op="inv")

  function _backward()
    x.grad += (-1 / x.data^2) * z.grad
  end
  z._backward = _backward

  z
end


function Base.:/(x::Union{Number, Value}, y::Union{Number, Value})::Value
  """ x / y = z
  """
  x = isa(x, Value) ? x : Value(x)
  y = isa(y, Value) ? y : Value(y)
  x * y^(-1)
end


function Base.:-(x::Value)::Value
  -1 * x
end


function relu(x::Union{Number, Value})::Value
  """ z = relu(x) = x > 0 ? x : 0
  """
  x = isa(x, Value) ? x : Value(x)
  z = Value(x.data < 0 ? 0 : x.data, _children=Set([x]), _op="relu")

  function _backward()
    x.grad += (x.data > 0 ? 1 : 0) * z.grad
  end
  z._backward = _backward

  z
end


function backward(x::Value)
  topo = Vector{Value}()
  visited = Set{Value}()
  function traverse_call_tree(v::Value)
    if !in(v, visited)
      push!(visited, v)
      for child in v._prev
        traverse_call_tree(child)
      end
      push!(topo, v)
    end
  end
  traverse_call_tree(x)

  x.grad = 1
  for v in Iterators.reverse(topo)
    v._backward()
  end
end

Base.show(io::IO, x::Value) = print(io, "Value(data=$(x.data), grad=$(x.grad))")
Base.show(io::IO, m::MIME"text/plain", x::Value) = print(io, "Value(data=$(x.data), grad=$(x.grad))")

end
end
