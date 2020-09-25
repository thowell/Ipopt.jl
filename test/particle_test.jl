# using Ipopt
include(joinpath(pwd(),"src/Ipopt.jl"))
using Test, ForwardDiff, LinearAlgebra

nc = 1
nf = 2
nq = 3
nu = 2
nβ = nc*nf
ns = 2

nx = nq+nu+nc+nβ+nc+2
np = nq+nβ+4nc

idx_s = nq+nu+nc+nβ+nc .+ (1:ns)

dt = 0.1

M(q) = 1.0*Matrix(I,nq,nq)
B(q) = [1. 0. 0.;0. 1. 0.]
P(q) = [1. 0. 0.;0. 1. 0.]

G(q) = [0; 0; 9.8]

N(q) = [0; 0; 1]

qpp = [0.,0.,0.1]
v0 = [10.,-7.0, 0.]
v1 = v0 - G(qpp)*dt
qp = qpp + 0.5*dt*(v0 + v1)

v2 = v1 - G(qp)*dt
q1 = qp + 0.5*dt*(v1 + v2)

qf = [0.; 0.; 0.]
uf = [0.; 0.]

W = 1.0*Matrix(I,nq,nq)
w = -W*qf
R = 1.0e-1*Matrix(I,nu,nu)
r = -R*uf
obj_c = 0.5*qf'*W*qf + 0.5*uf'*R*uf

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = x[nq+nu+nc]
    β = x[nq+nu+nc .+ (1:nβ)]
    ψ = x[nq+nu+nc+nβ+nc]
	s = x[nq+nu+nc+nβ+nc .+ (1:ns)]

    return q,u,y,β,ψ,s
end

function obj(x)
    q,u,y,β,ψ,s = unpack(x)
    return 0.5*q'*W*q + w'*q + 0.5*u'*R*u + r'*u + obj_c + λ_global'*s + 0.5*mu_prev*s'*s
end

function con!(c,x)
    q,u,y,β,ψ,s = unpack(x)

    c[1] = (N(q)'*q)
    c[2] = (((0.5*y)^2 - β'*β))
    c[3:5] = (M(q)*(2*qp - qpp - q)/dt - G(q)*dt + B(q)'*u + P(q)'*β + N(q)*y)
    c[6:7] = (P(q)*(q-qp)/dt + 2.0*β*ψ)
    c[8] = s[1] - y*(N(q)'*q)
    c[9] = s[2] - ψ*((0.5*y)^2 - β'*β)

	return c
end

n = nx
m = np

x_L = zeros(nx)
x_L[1:(nq+nu)] .= -1.0e8
x_L[nq+nu+nc .+ (1:nβ)] .= -1.0e8
x_U = 1.0e8*ones(nx)
cI_idx = zeros(Bool,m)
cI_idx[1:nc+nc] .= 1

g_L = zeros(m)
g_U = zeros(m)
g_U[1:2nc] .= 1.0e8

q0 = q1
u0 = 1.0e-8*ones(nu)
y0 = 1.0e-8*ones(1)[1]
β0 = 1.0e-8*ones(nβ)
ψ0 = 1.0e-8*ones(1)[1]
s0 = 1.0e-8*ones(2)
x0 = [q0;u0;y0;β0;ψ0;s0]

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

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
    row_col!(row,col,(1:m),(1:n))
		rows .= vec(row)
		cols .= vec(col)
  else

  	values .= vec(ForwardDiff.jacobian(con!,zeros(m),x))
  end
end

function convergence(x,tol)
	q,u,y,β,ψ,s = unpack(x)

	cond1 = norm(s,Inf) < tol
    c_tmp = zeros(m)
	con!(c_tmp,x_global)
	c_tmp[1:2nc] .= min.(0.0,c_tmp[1:2:nc])
	cond2 = norm(c_tmp,Inf) < tol

	return cond1 && cond2
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
    λ_global .+= 1/mu_prev*x_global[idx_s]
    global mu_prev = mu
  end

  return true#!convergence(x_global,1.0e-4) #iter_count < 1  # Interrupts after one iteration.
end

function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
  if mode == :Structure

  else

  end
end

prob = createProblem(n, x_L, x_U, m, g_L, g_U, m*n, n*n,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, )
addOption(prob,"hessian_approximation","limited-memory")
addOption(prob,"mu_strategy","monotone")

addOption(prob,"constr_viol_tol",1.0e-3)
addOption(prob,"tol",1.0e-3)
# addOption(prob,"")
setIntermediateCallback(prob, intermediate)

global x_global = zeros(n)
global λ_global = zeros(ns)
global mu_prev = 0.1

prob.x .= 1.0e-5*rand(n)
prob.x[1:nq] = copy(q1)
x_global .= prob.x
@time solvestat = solveProblem(prob)

λ_global
mu_prev
# @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop

prob.x[idx_s]
