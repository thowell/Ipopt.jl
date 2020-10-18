using LinearAlgebra, ForwardDiff

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

v = [5.0; 5.0; 0.]
n = 1.0
μ = 0.5

P = [1. 0. 0.;
	 0. 1. 0.]

_norm(x) = sqrt(x'*x)
function vec_norm(x)
	if norm(x) == 0.0
		return ones(length(x))./norm(ones(length(x)))
	else
		x./_norm(x)
	end
end

function d_vec_norm(x)
	if norm(x) == 0.0
		y = 1.0*ones(length(x))#./norm(ones(length(x)))
		return (I - y*y'/(y'*y))/_norm(y)
	else
		(I - x*x'/(x'*x))/_norm(x)
	end
end
b = 1.0e-3*ones(2)
b = zeros(2)
d_vec_norm(b)
_norm(b)
vec_norm(b)
d_vec_norm(b)
norm(vec(d_vec_norm(b)) - vec(ForwardDiff.jacobian(vec_norm,norm(b) == 0.0 ? 1.0*ones(2) : b)))

ψ = 0.1
P*v + ψ*vec_norm(b)

##
include(joinpath(pwd(),"src/Ipopt.jl"))

nx = 4
nc = 2 + 1 + 1
function obj(x)
	b = x[1:2]
	ψ = x[3]
	s = x[4]

	return λ_global[1]*s + 0.5/mu_prev*s^2
end

function con!(c,x)
	b = x[1:2]
	ψ = x[3]
	s = x[4]

	c[1:2] = P*v + ψ*vec_norm(b)
	c[3] = μ*n - _norm(b)
	c[4] = s - min(ψ,(μ*n - _norm(b)))

	return nothing
end

function ∇con!(C,x)
	b = x[1:2]
	ψ = x[3]

	C[1:2,1:2] = ψ*d_vec_norm(b)
	C[1:2,3] = vec_norm(b)

	C[3,1:2] = -1.0*vec_norm(b)
	C[3,3] = 0.0

	if ψ < μ*n - _norm(b)
		C[4,3] = -1.0
	end

	if ψ > μ*n - _norm(b)
		C[4,1:2] = vec_norm(b)
	end

	C[4,4] = 1.0
	return nothing
end

_x = [1.0e-3; 1.0e-3; 1.0e-3; 1.0e-3]
obj(_x)
_c = zeros(nc)
con!(_c,_x)
_c
_C = zeros(nc,nx)
∇con!(_C,_x)

function eval_f(x::Vector{Float64})
	x_global .= x
  return obj(x)
end

function eval_g(x::Vector{Float64}, g::Vector{Float64})
	con!(g,x)

	return g
end

function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
	grad_f .= ForwardDiff.gradient(obj,x)
end

function eval_jac_g(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
	if mode == :Structure
		row = []
		col = []
		row_col!(row,col,(1:nc),(1:nx))
		rows .= vec(row)
		cols .= vec(col)
	else
		∇con!(reshape(values,nc,nx),x)
	end
end

function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
	if mode == :Structure
		nothing
	else
		nothing
	end
end

# This tests callbacks.
function intermediate(alg_mod::Int, iter_count::Int,
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  ls_trials::Int)

	println("CALLBACK")
  println("   mu_prev: $(mu_prev)")
  println("   mu: $(mu)")
  println("   x: $(x_global)")

  if mu != mu_prev
    println("-mu difference-")
    println("-AL update-")
    λ_global[1] += 1/mu_prev*x_global[4]
    global mu_prev = mu
  end

  return true#!(norm(x_global[9:11],Inf) < 1.0e-3 && norm(eval_g(x_global,zeros(7))) < 1.0e-3) #iter_count < 1  # Interrupts after one iteration.
end

x_L = [-1.0e8; -1.0e8; 0.0; 0.0]
x_U = 1.0e8*ones(nx)

g_L = zeros(nc)
g_U = [zeros(2);1.0e8;0.0]

prob = Ipopt.createProblem(nx, x_L, x_U, nc, g_L, g_U, nc*nx, nx*nx,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
Ipopt.addOption(prob,"hessian_approximation","limited-memory")
# Ipopt.addOption(prob,"mu_strategy","monotone")

Ipopt.addOption(prob,"constr_viol_tol",1.0e-3)
Ipopt.addOption(prob,"tol",1.0e-3)
Ipopt.setIntermediateCallback(prob, intermediate)

global x_global = zeros(nx)
global λ_global = zeros(1)
global mu_prev = 0.1

prob.x .= 1.0e-3*rand(nx)
x_global .= prob.x
@time solvestat = Ipopt.solveProblem(prob)

b_sol = prob.x[1:2]
ψ_sol = prob.x[3]
s_sol = prob.x[4]
λ_global
mu_prev

norm(v[1:2])
vec_norm(v[1:2])'*vec_norm(b_sol)

con!(_c,prob.x)
_c
_C = zeros(nc,nx)
∇con!(_C,prob.x)
_C
