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

v = [10.; 1.; 0.]
n = 0.0
μ = 0.5

P = [1. 0. 0.;
	 0. 1. 0.]

b = 1.0e-8*ones(2)
_norm(x) = sqrt(x'*x)
function vec_norm(x)
	if norm(x) < 1.0e-8
		return ones(length(x))./norm(ones(length(x)))
	else
		x./_norm(x)
	end
end

function d_vec_norm(x)
	if norm(x) == 0.0
		y = ones(length(x))#./norm(ones(length(x)))
		return (I - y*y'/(y'*y))/_norm(y)
	else
		(I - x*x'/(x'*x))/_norm(x)
	end
end

d_vec_norm(b)
_norm(b)
vec_norm(b)
d_vec_norm(b)
norm(vec(d_vec_norm(b)) - vec(ForwardDiff.jacobian(vec_norm,all(b .== 0.0) ? ones(2) : b)))

ψ = 0.1
P*v + ψ*vec_norm(b)

##
include(joinpath(pwd(),"src/Ipopt.jl"))

nx = 3
nc = 2 + 1 + 1
function obj(x)
	b = x[1:2]
	ψ = x[3]

	0.0
end

function con!(c,x)
	b = x[1:2]
	ψ = x[3]

	c[1:2] = P*v + ψ*vec_norm(b)
	c[3] = μ*n - _norm(b)
	c[4] = ψ*(μ*n - _norm(b))

	return nothing
end

function ∇con!(C,x)
	b = x[1:2]
	ψ = x[3]

	C[1:2,1:2] = ψ*d_vec_norm(b)
	C[1:2,3] = vec_norm(b)

	C[3,1:2] = -1.0*vec_norm(b)
	C[3,3] = 0.0

	C[4,1:2] = -1.0*ψ*vec_norm(b)
	C[4,3] = (μ*n - _norm(b))

	return nothing
end

_x = [1.0e-3; 1.0e-3; 1.0e-5]
obj(_x)
_c = zeros(nc)
con!(_c,_x)
_c
_C = zeros(nc,nx)
∇con!(_C,_x)
_C


function eval_f(x::Vector{Float64})
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

x_L = [-1.0e8; -1.0e8; 0.0]
x_U = 1.0e8*ones(nx)

g_L = zeros(nc)
g_U = [zeros(2);1.0e8;0.0]

prob = Ipopt.createProblem(nx, x_L, x_U, nc, g_L, g_U, nc*nx, nx*nx,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
Ipopt.addOption(prob,"hessian_approximation","limited-memory")
Ipopt.addOption(prob,"mu_strategy","monotone")

Ipopt.addOption(prob,"constr_viol_tol",1.0e-3)
Ipopt.addOption(prob,"tol",1.0e-3)

prob.x .= 1.0e-5*rand(nx)
@time solvestat = Ipopt.solveProblem(prob)

b_sol = prob.x[1:2]
ψ_sol = prob.x[3]

vec_norm(v[1:2])'*vec_norm(b_sol)

_C = zeros(nc,nx)
∇con!(_C,prob.x)
_C
