# Risk-averse newsvendor
# ----------------------
using JuMP
using Clp
using Distributions
using Random
using Plots

include("benders.jl")


function sim_scenarios(Ns)
    Random.seed!(123)
    d = Uniform(60,150)
    # Ns sample for each replica
    S = rand(d, Ns)
    return S
end

# 1) Compute and plot the function \mathcal{Q}(x)=E[Q(x,\tilde{d})] for x\in\{60,70,\ldots,150\}.
function compute_Q(D)
    c=10
    r=5
    q=25
    u = 150
    X = collect(60:10:150)
    Qx = []
    for x in X
        value = news_vendor_2nd_st(D,c,q,r,u,x)
        push!(Qx, value)
    end
    return Qx, X
end

# first stage problem
function first_stage(benders::BendersAlgorithm)
    c=10
    u = 150

    m = JuMP.Model(Clp.Optimizer)
    @variable(m, 0<=x<=u)
    ncen = 1
    if benders.avgcut
        @variable(m, z)
    else
        ncen = length(benders.scen_probability)
        @variable(m, z[i=1:ncen])
    end

    if length(benders.cuts) > 0
        add_cuts!(m, benders)
        if benders.avgcut
            @objective(m, Min, c*x + z)
        else
            @objective(m, Min, c*x + sum(z[i] * benders.scen_probability[i] for i in 1:ncen))
        end
    else
        @objective(m, Min, c*x)
    end

    print(m)
    optimize!(m)
    termination_status(m)
    value.(z)
    value(x)

    if benders.avgcut
        zbar = value(z)
    else
        zbar = sum(value(z[i]) * benders.scen_probability[i] for i in 1:ncen)
    end
    return firstStageSolution(value(x), zbar, objective_value(m))
end

function second_stage(x::Float64, d::Float64)
    r = -5
    q = -25
    h = 0
    T = -1

    m = JuMP.Model(Clp.Optimizer)
    @variable(m, y>=0)
    @variable(m, w>=0)
    @constraint(m, Ressource, y + w + T*x == h)
    @constraint(m, y <= d)

    @objective(m, Min, q*y + r*w)
    optimize!(m)
    termination_status(m)

    return secondStageSolution(value(y),  value(w), dual(Ressource), objective_value(m), h, T)
end


D = sim_scenarios(100)
Qx, X = compute_Q(D)



# a) Model, reformulate (as a deterministic equivalent) and solve the newsvendor problem using the Conditional Value-at-Risk instead of the expected value.
function news_vendor_det_eq(d, c, q, r, u, alpha, prob)
    Ns = length(d)

    m = JuMP.Model(with_optimizer(Clp.Optimizer))
    @variable(m, 0<=x<=u)
    @variable(m, y[1:Ns]>=0)
    @variable(m, w[1:Ns]>=0)

    @constraint(m, [s=1:Ns], y[s] + w[s] <= x)
    @constraint(m, [s=1:Ns], y[s] <= d[s])

    # 2nd stage cost
    @expression(m, values[s=1:Ns], q*y[s] + r*w[s])

    # CVAR Model
    @variable(m, var)
    @variable(m, delta[s=1:Ns] >= 0)
    @constraint(m, constraint1[s=1:Ns], delta[s] + var >= values[s])

    @expression(m, cvar_obj, var + sum(prob[s] * delta[s] / (1 - alpha) for s in 1:Ns) )

    @objective(m, Min, c*x + cvar_obj)
    optimize!(m)
    termination_status(m)
    return objective_value(m)
end
c=10.0
q=-25.0
r=-5.0
u=150
ncen = length(D)
prob = 1/ncen * ones(ncen)
cost1 = news_vendor_det_eq(D, c, q, r, u, alpha, prob)
# -1428.5053275079806

# cost1 = news_vendor_det_eq(D, c, q, r, u, 0, prob)
# stage1 = benders.solutions[end].first_stage
# stage2 = benders.solutions[end].second_stage
# total_cost(benders, stage1, stage2)

# b) Show how the first stage solution of the newsvendor varies with the confidence level \alpha of the Conditional Value-at-Risk.

# c) Implement the Benders decomposition for the CV@R-based newsvendor using the CV@R dual representation
alpha = 0.05
avgcut = false
benders, it = benders_algorithm_CVaR(first_stage, second_stage, D, alpha, avgcut)
stage1 = benders.solutions[end].first_stage
stage2 = benders.solutions[end].second_stage
total_cost(benders, stage1, stage2)