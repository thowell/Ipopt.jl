# using Ipopt
include(joinpath(pwd(),"src/Ipopt.jl"))
using Test, ForwardDiff

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

obj(z) = (z[1] - 5)^2 + (2*z[2] + 1)^2 + λ_global'*z[9:11] + 0.5*mu_prev*z[9:11]'*z[9:11]

function con!(g,z)
	g[1] = 2*(z[2] - 1) - 1.5*z[2] + z[3] - 0.5*z[4] + z[5]
	g[2] = 3*z[1] - z[2] - 3.0 - z[6]
	g[3] = -z[1] + 0.5*z[2] + 4.0 - z[7]
	g[4] = -z[1] - z[2] + 7.0 - z[8]
	g[5] = z[9] - z[3]*z[6]
	g[6] = z[10] - z[4]*z[7]
	g[7] = z[11] - z[5]*z[8]

	return g
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

  	values .= vec(ForwardDiff.jacobian(con!,zeros(7),x))
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
    λ_global .+= 1/mu_prev*x_global[9:11]
    global mu_prev = mu
  end

  return true#!(norm(x_global[9:11],Inf) < 1.0e-3 && norm(eval_g(x_global,zeros(7))) < 1.0e-3) #iter_count < 1  # Interrupts after one iteration.
end

n = 11
m = 7
x_L = zeros(n)
x_U = 1.0e8*ones(n)

g_L = zeros(m)
g_U = zeros(m)
# g_U[5:7] .= 1.0e8

prob = createProblem(n, x_L, x_U, m, g_L, g_U, m*n, n*n,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
addOption(prob,"hessian_approximation","limited-memory")
addOption(prob,"mu_strategy","monotone")

addOption(prob,"constr_viol_tol",1.0e-3)
addOption(prob,"tol",1.0e-3)# addOption(prob,"")
setIntermediateCallback(prob, intermediate)

global x_global = zeros(n)
global λ_global = zeros(3)
global mu_prev = 0.1

prob.x .= 1.0e-5*rand(n)
x_global .= prob.x
@time solvestat = solveProblem(prob)

λ_global
mu_prev
# @test Ipopt.ApplicationReturnStatus[solvestat] == :User_Requested_Stop

prob.x[9:11]
