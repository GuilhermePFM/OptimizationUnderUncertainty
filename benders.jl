# Benders decomposition for the newsvendor problem
using JuMP
using Clp
using Distributions
using Random
using Plots

# 2) Develop a Benders decomposition algorithm for the newsvendor problem described above. In the same plot of item 1, plot the piecewise linear approximation for each iteration of the method (one plot for each iteration containing the true function and the current approximation.
mutable struct firstStageSolution
    x
    zbar
    obj_value
end
mutable struct secondStageSolution
    y
    w
    pbar
    Qx
    h
    T
end
mutable struct Solution
    first_stage::firstStageSolution
    second_stage::Vector{secondStageSolution}
end
abstract type AbstractCut end
mutable struct Cut <:AbstractCut
    cte::Float64 # b (cte)
    ang_coef::Float64 # a (angular coefficient)
end
mutable struct AvgCut <:AbstractCut
    cte::Float64 # b (cte)
    ang_coef::Float64 # a (angular coefficient)
end

import Base
Base.@kwdef mutable struct BendersAlgorithm
    UB::Float64
    LB::Float64
    ϵ::Float64

    first_stage
    second_stage
    c::Vector{Float64}
    q::Vector{Float64}
    r::Vector{Float64}
    scenarios::Vector{Float64}

    solutions::Array{Solution}

    best_solution

    avgcut::Bool
    cuts::Dict{Int64,Vector{Cut}}
    orig_probability::Vector{Float64}
    scen_probability::Vector{Float64}

    BendersAlgorithm(e::Float64, first_stage, second_stage, c::Vector{Float64}, q::Vector{Float64}, r::Vector{Float64}, probability) = new(Inf, -Inf, e, first_stage, second_stage, c, q, r, [], [], (0,0), true, Dict(), probability, probability)
    BendersAlgorithm(UB, LB, e, first_stage, second_stage, c, q, r, scenarios, solutions, best_solution, avgcut, cuts, probability) = new(UB, LB, e, first_stage, second_stage, c, q, r, scenarios, solutions, best_solution, avgcut, cuts, probability, probability)
end

function total_cost(benders, first_stage::firstStageSolution, second_stage::Vector{secondStageSolution})

    first_stage_cost = sum(benders.c' * first_stage.x)

    # second_stage_cost = sum(benders.q' *i.y + benders.r' * i.w for i in last_sol.second_stage) / length(last_sol.second_stage)
    Eqx = sum(sol2.Qx * benders.scen_probability[i] for (i,sol2) in enumerate(second_stage))

    total_expected_cost = first_stage_cost + Eqx

    return total_expected_cost
end

function cut_algorithm!(benders::BendersAlgorithm, first_stage::firstStageSolution, second_stage::secondStageSolution, s)
    # elseif second_stage.Qx == first_stage.zbar
    #     # c) check theorical convergence
    #     # end here
    #     # theoretical convergence
    #     print("theoretical convergence")
    #     conv.theoretical = true
    if second_stage.Qx == Inf
        # step 4 add ray cut (feasibility)
        get_feasibility_cut!(benders, second_stage)
    else
        # add cut 
        get_optimality_cut!(benders, first_stage, second_stage, s)
    end

    return nothing
end

function get_optimality_cut!(benders, first_stage::firstStageSolution, second_stage::secondStageSolution, s)
    dual = second_stage.pbar
    ang_coef = - second_stage.T*dual
    cte = second_stage.Qx - ang_coef*first_stage.x
    cut = Cut(cte, ang_coef)

    if s in keys(benders.cuts)
        push!(benders.cuts[s], cut)
    else
        benders.cuts[s] = [cut]
    end

    return nothing
end

function get_feasibility_cut!(benders, second_stage::secondStageSolution)
   #todo
end

function get_avg_cut(benders::BendersAlgorithm, c::I) where I<:Number
    nscen = length(keys(benders.cuts))
    ncut = length(benders.cuts[nscen])
    cte = 0
    ang_coef = 0
    for s in 1:nscen
        cut = benders.cuts[s][c]
        cte += cut.cte / nscen
        ang_coef += cut.ang_coef / nscen
    end
    return AvgCut(cte, ang_coef)
end

function add_cut!(m, cut::T) where T<: AbstractCut
    x = m[:x]
    z = m[:z]
    @constraint(m, cut.cte + cut.ang_coef * x <= z)
end

function add_cut!(m, cut::T, k) where T<: AbstractCut
    x = m[:x]
    z = m[:z]
    @constraint(m, cut.cte + cut.ang_coef * x <= z[k])
end

function add_cuts!(m, benders::BendersAlgorithm)
    if benders.avgcut
        ncut = length(benders.cuts[1])
        for c in 1:ncut
            cut = get_avg_cut(benders, c)
            add_cut!(m, cut)
        end
    else
        for k in keys(benders.cuts)
            for cut in benders.cuts[k]
                add_cut!(m, cut, k)
            end
        end
    end
    return nothing
end

function update!(benders::BendersAlgorithm, sol::Solution)
    # update vector of solutions
    push!(benders.solutions, sol)

    # calculate total expected cost
    total_expected_cost = total_cost(benders, sol.first_stage, sol.second_stage)

    # update UB and best solution
    if total_expected_cost < benders.UB
        benders.UB = total_expected_cost
        benders.best_solution = (sol.first_stage.x, sol.second_stage)
    end
    return nothing
end

function get_conv(benders::BendersAlgorithm)
    conv = abs(benders.UB - benders.LB) <= benders.ϵ

    return conv
end

function benders_algorithm(first_stage, second_stage, D, avgcut=true)
    # step 1
    # UB = Inf
    # LB = -Inf
    e = 1e-3
    c=10.0
    q=-25.0
    r=-5.0
    ncen = length(D)
    probability = [1/ncen for i in 1:ncen]
    benders = BendersAlgorithm(Inf, -Inf, e, first_stage, second_stage, [c], [q], [r], [], [], (0,0), avgcut, Dict(), probability)
    # benders = BendersAlgorithm(e, first_stage, second_stage, [c], [q], [r])
    benders.scenarios = D
    maxit = 10
    for i in 1:maxit
         # step 2
         sol1 = benders.first_stage(benders)
         benders.LB = sol1.obj_value

         # step 3
         # a)
         scenario_sol = []
         for (s,d) in enumerate(benders.scenarios)
             sol2 = benders.second_stage(sol1.x, d)
 
             # b) if Q(x) > zbar, update upper bound
             cut_algorithm!(benders, sol1, sol2, s)
             push!(scenario_sol, sol2)
        end
         # update best solution
         update!(benders, Solution(sol1, scenario_sol))

        # check convergence
        converged = false
        if i > 1
            converged = get_conv(benders)
        end

        if converged 
            print("Benders converged in $i iterations")
            return benders, i
        end
    end
end

function CVaR(values, prob, alpha)
    ncen = length(values)

    m = JuMP.Model(Clp.Optimizer)
    @variable(m, var)
    @variable(m, delta[s=1:ncen] >= 0)
    @constraint(m, constraint1[s=1:ncen], delta[s] + var >= values[s])

    @objective(m, Min, var + sum(prob[s] * delta[s] / (1 - alpha) for s in 1:ncen) )
    optimize!(m)

    return objective_value(m), value(var), dual.(constraint1)
end

function benders_algorithm_CVaR(first_stage, second_stage, D, alpha, avgcut=true)
    # step 1
    # UB = Inf
    # LB = -Inf
    e = 1e-3
    c=10.0
    q=-25.0
    r=-5.0
    ncen = length(D)
    probability = [1/ncen for i in 1:ncen]
    benders = BendersAlgorithm(Inf, -Inf, e, first_stage, second_stage, [c], [q], [r], [], [], (0,0), avgcut, Dict(), probability)
    # benders = BendersAlgorithm(e, first_stage, second_stage, [c], [q], [r])
    benders.scenarios = D
    maxit = 10
    for i in 1:maxit
         # step 2
         sol1 = benders.first_stage(benders)
         benders.LB = sol1.obj_value

         # step 3
         # a)
         scenario_sol = []
         for (s,d) in enumerate(benders.scenarios)
             sol2 = benders.second_stage(sol1.x, d)
 
             # b) if Q(x) > zbar, update upper bound
             cut_algorithm!(benders, sol1, sol2, s)
             push!(scenario_sol, sol2)
        end
        # calculates CVaR
        values = [sol.Qx for sol in scenario_sol]
        cvar, var, cvar_probabilities = CVaR(values, benders.orig_probability, alpha)
        benders.scen_probability = cvar_probabilities

         # update best solution
         update!(benders, Solution(sol1, scenario_sol))

        # check convergence
        converged = false
        if i > 1
            converged = get_conv(benders)
        end

        if converged 
            print("Benders converged in $i iterations")
            return benders, i
        end
    end
end