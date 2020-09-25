# using Ipopt
include(joinpath(pwd(),"src/Ipopt.jl"))
using Test, ForwardDiff, LinearAlgebra

function linear_interp(x0,xf,T)
    n = length(x0)
    X = [copy(Array(x0)) for t = 1:T]

    for t = 1:T
        for i = 1:n
            X[t][i] = (xf[i]-x0[i])/(T-1)*(t-1) + x0[i]
        end
    end

    return X
end

mutable struct Hopper{T,S}
    mb::T
    ml::T
    Jb::T
    Jl::T
    r::T
    μ::T
    g::T
    k::S
    Δt::T
end

# Dimensions
nq = 5 # configuration dim
nu = 2 # control dim
nc = 1 # number of contact points
nf = 1 # number of faces for friction cone pyramid
nβ = nc*nf
ns = 2

nx = nq+nu+nc+nβ+nc+ns
np = nq+nβ+4nc
T = 10 # number of time steps to optimize

# Parameters
g = 9.81 # gravity
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
mb = 10. # body mass
ml = 1.  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

# Kinematics
r = 0.7
p1(q) = [q[1] + q[3]*sin(q[5]), q[2] - q[3]*cos(q[5])]

# Methods
M(h::Hopper,q) = Diagonal([h.mb+h.ml, h.mb+h.ml, h.ml, h.Jb, h.Jl])
∇V(h::Hopper,q) = [0., (h.mb+h.ml)*h.g, 0., 0., 0.]

C(h::Hopper,qk,qn) = zeros(nq)

function ϕ(::Hopper,q)
    q[2] - q[3]*cos(q[5])
end

N(::Hopper,q) = ([0., 1., -cos(q[5]), 0., q[3]*sin(q[5])])'

function P(::Hopper,q)
        ([1., 0., sin(q[5]), 0., q[3]*cos(q[5])])'
end

B(::Hopper,q) = [0. 0. 0. 1. -1.;
                 0. 0. 1. 0. 0.]

model = Hopper(mb,ml,Jb,Jl,r,μ,g,p1,Δt)

function unpack(x)
    q = x[1:nq]
    u = x[nq .+ (1:nu)]
    y = nc == 1 ? x[nq+nu+nc] : x[nq+nu .+ (1:nc)]
    β = nβ == 1 ? x[nq+nu+nc+nβ] : x[nq+nu+nc .+ (1:nβ)]
    ψ = nc == 1 ? x[nq+nu+nc+nβ+nc] : x[nq+nu+nc+nβ .+ (1:nc)]
    s = x[nq+nu+nc+nβ+nc .+ (1:ns)]

    return q,u,y,β,ψ,s
end


W = Diagonal([1.0,1.0,1.0,1.0e-1,1.0e-1])
R = Diagonal([1.0e-1,1.0e-3])
Wf = Diagonal(5.0*ones(nq))
q0 = [0., r, r, 0., 0.]
qf = [2.0, r, r, 0., 0.]
q0_ref = [0., model.r, 0.5*model.r, 0., 0.]
qf_ref = [2.0, model.r, 0.5*model.r, 0., 0.]
Q0_ref = [q0,linear_interp(q0_ref,qf_ref,T+1)[1:end-1]...,qf]
uf = zeros(nu)
w = -W*qf
wf = -Wf*qf
rr = -R*uf
obj_c = [0.5*(Q0_ref[t+2]'*W*Q0_ref[t+2] + uf'*R*uf) for t = 1:T]
obj_cf = [0.5*(Q0_ref[t+2]'*Wf*Q0_ref[t+2] + uf'*R*uf) for t = 1:T]

Q0 = linear_interp(q0,qf,T+2)

qpp = Q0[2]
qp = Q0[2]

function obj(z)
    _sum = 0.
    for t = 1:T
        q,u,y,β,ψ,s = unpack(z[(t-1)*nx .+ (1:nx)])

        if t != T
            _sum += 0.5*q'*W*q + w'*q + 0.5*u'*R*u + rr'*u + obj_c[t] + λ_global[(t-1)*ns .+ (1:ns)]'*s + 0.5*mu_prev*s'*s
        else
            _sum += 0.5*q'*Wf*q + wf'*q + 0.5*u'*R*u + rr'*u + obj_cf[t] + λ_global[(t-1)*ns .+ (1:ns)]'*s + 0.5*mu_prev*s'*s
        end
    end
    return _sum
end

function con!(c,z)

    for t = 1:T
        q,u,y,β,ψ,s = unpack(z[(t-1)*nx .+ (1:nx)])

        if t == 1
            _qpp = qpp
            _qp = qp
        elseif t == 2
            _qpp = qp
            _qp = z[(t-2)*nx .+ (1:nq)]
        else
            _qpp = z[(t-3)*nx .+ (1:nq)]
            _qp = z[(t-2)*nx .+ (1:nq)]
        end

        c[(t-1)*np .+ (1:np)] .= [(ϕ(model,q));
                                  ((model.μ*y)^2 - β'*β);
                                  (1/model.Δt*(M(model,_qpp)*(_qp - _qpp) - M(model,_qp)*(q - _qp)) - model.Δt*∇V(model,_qp) + B(model,q)'*u +  N(model,q)'*y + P(model,q)'*β);
                                  (P(model,q)*(q-_qp)/model.Δt + 2.0*β*ψ);
                                  s[1] - ϕ(model,q)*y;
                                  s[2] - ((model.μ*y)^2 - β'*β)*ψ]
     end
     return c
end

n = T*nx
m = T*np
x_L = zeros(T*nx)
x_U = 1.0e8*ones(T*nx)
for t = 1:T
    x_L[(t-1)*nx .+ (1:nq+nu)] .= -1.0e8
    x_L[(t-1)*nx + 3] = model.r/2.
    x_U[(t-1)*nx + 3] = model.r
    x_L[(t-1)*nx+nq+nu+nc .+ (1:nβ)] .= -1.0e8
    # x_L[(t-1)*nx+nq+nu+nc+nβ+nc .+ (1:ns)] .= -1.0e8
end
# cI_idx_t = zeros(Bool,np)
# cI_idx_t[1:nc+nc] .= 1
# cI_idx = zeros(Bool,m)
# for t = 1:T
#     cI_idx[(t-1)*np .+ (1:np)] .= cI_idx_t
# end

g_L = zeros(m)
g_U = zeros(m)
for t = 1:T
    g_U[(t-1)*np .+ (1:2nc)] .= 1.0e8
    # g_U[(t-1)*np + nq+nβ+2nc .+ (1:2nc)] .= 1.0e8
end

u0 = 1.0e-5*ones(nu)
y0 = 1.0e-5*ones(1)[1]
β0 = 1.0e-5*ones(nβ)[1]
ψ0 = 1.0e-5*ones(1)[1]
s0 = 1.0e-5*ones(ns)

x0 = zeros(T*nx)
for t = 1:T
    x0[(t-1)*nx .+ (1:nx)] .= [Q0[t+2];u0;y0;β0;ψ0;s0]
end

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
	s_tmp = zeros(ns*T)
	for t = 1:T
		q,u,y,β,ψ,s = unpack(x[(t-1)*nx .+ (1:nx)])
		s_tmp[(t-1)*ns .+ (1:ns)] = copy(s)
	end

	cond1 = norm(s_tmp,Inf) < tol

    c_tmp = zeros(m)
	con!(c_tmp,x_global)
	for t = 1:T
		c_tmp[(t-1)*np .+ (1:2nc)] .= min.(0.0,c_tmp[(t-1)*np .+ (1:2:nc)])
	end
	cond2 = norm(c_tmp,Inf) < tol

	return cond1 && cond2
end
convergence(x0,1.0)

# This tests callbacks.
function intermediate(alg_mod::Int, iter_count::Int,
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64,
  ls_trials::Int)

	# println("CALLBACK")
  # println("   mu_prev: $(mu_prev)")
  # println("   mu: $(mu)")
  # println("   x: $(x_global)")

  if mu != mu_prev
    println("-mu difference-")
    println("-AL update-")
	for t = 1:T
		q,u,y,β,ψ,s = unpack(x_global[(t-1)*nx .+ (1:nx)])

    	λ_global[(t-1)*ns .+ (1:ns)] .+= 1/mu_prev*s
	end
    global mu_prev = mu
  end

  return true#!convergence(x_global,1.0e-3) #iter_count < 1  # Interrupts after one iteration.
end

function eval_h(x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, obj_factor::Float64, lambda::Vector{Float64}, values::Vector{Float64})
  if mode == :Structure

  else

  end
end

prob = createProblem(n, x_L, x_U, m, g_L, g_U, m*n, n*n,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
addOption(prob,"hessian_approximation","limited-memory")
addOption(prob,"mu_strategy","monotone")

addOption(prob,"constr_viol_tol",1.0e-2)
addOption(prob,"tol",1.0e-2)

# addOption(prob,"")
setIntermediateCallback(prob, intermediate)

global x_global = zeros(n)
global λ_global = zeros(ns*T)
global mu_prev = 0.1

prob.x .= copy(x0)
x_global .= prob.x
@time solvestat = solveProblem(prob)

λ_global
mu_prev
# @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop

convergence(prob.x,1.0e-3)
for t = 1:T
	for t = 1:T
		q,u,y,β,ψ,s = unpack(x_global[(t-1)*nx .+ (1:nx)])
		println("s: $(s)")
	end
end
