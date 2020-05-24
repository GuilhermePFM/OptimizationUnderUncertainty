# Benders decomposition for the newsvendor problem
using JuMP
using Clp
using Distributions
using Random
using Plots

function news_vendor_2nd_st(d,c,q,r,u,x)
    Ns = length(d)

    m = JuMP.Model(Clp.Optimizer)
    @variable(m, y[1:Ns]>=0)
    @variable(m, w[1:Ns]>=0)
    @constraint(m, [s=1:Ns], y[s] + w[s] <= x)
    @constraint(m, [s=1:Ns], y[s] <= d[s])

    @objective(m, Min, - 1/Ns*sum(q*y[s] + r*w[s] for s in 1:Ns))
    optimize!(m)
    termination_status(m)
    return objective_value(m)
end

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

# D = sim_scenarios(1000)
# Qx, X = compute_Q(D)
# plot(X, Qx, title = "Question 1) compute and plot the function Q(x)", seriestype = :scatter)
# xlabel!("x")
# ylabel!("Q(x)")

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

    BendersAlgorithm(e::Float64, first_stage, second_stage, c::Vector{Float64}, q::Vector{Float64}, r::Vector{Float64}) = new(Inf, -Inf, e, first_stage, second_stage, c, q, r, [], [], (0,0), true, Dict() )
    BendersAlgorithm(UB, LB, e, first_stage, second_stage, c, q, r, scenarios, solutions, best_solution, avgcut, cuts) = new(UB, LB, e, first_stage, second_stage, c, q, r, scenarios, solutions, best_solution, avgcut, cuts)
end

mutable struct Convergence 
    theoretical::Bool
    computational::Bool
    Convergence() = new(false, false)
end
function total_cost(first_stage::firstStageSolution, second_stage::Vector{secondStageSolution})

    first_stage_cost = sum(benders.c' * first_stage.x)

    # second_stage_cost = sum(benders.q' *i.y + benders.r' * i.w for i in last_sol.second_stage) / length(last_sol.second_stage)
    Eqx = sum(i.Qx for i in second_stage) / length(second_stage)

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
    total_expected_cost = total_cost(sol.first_stage, sol.second_stage)

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
    benders= BendersAlgorithm(Inf, -Inf, e, first_stage, second_stage, [c], [q], [r], [], [], (0,0), avgcut, Dict())
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

# first stage problem
function first_stage(benders::BendersAlgorithm)
    c=10
    u = 150

    m = JuMP.Model(Clp.Optimizer)
    @variable(m, 0<=x<=u)
    ncut = 1
    if benders.avgcut
        @variable(m, z)
    else
        ncut = length(keys(benders.cuts))
        @variable(m, z[i=1:ncut])
    end

    if length(benders.cuts) > 0
        add_cuts!(m, benders)
        @objective(m, Min, c*x + sum(z)/ncut)
    else
        @objective(m, Min, c*x)
    end

    print(m)
    optimize!(m)
    termination_status(m)
    value.(z)
    value(x)
    zbar = sum(value.(z))/ncut
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
benders, it = benders_algorithm(first_stage, second_stage,D, false)

# graph = plot(X, Qx, title = "Question 2) Benders on Q(x)")
# xlabel!("x")
# ylabel!("Q(x)")

# Total cost
# Tx = Qx .+ 10*X
# graph = plot(X, Tx, title = "Question 2) Benders on T(x)")
# xlabel!("x")
# ylabel!("T(x)")
# tmin, i = findmin(Tx)
# xmin = X[i]
# sum(D)/100
# for i in 1:7
#     newcut = get_avg_cut(benders, i)
#     cut_line = [newcut.cte + newcut.ang_coef*x for x in X]
#     plot!(graph, X, cut_line)
# end
# display(graph)

function plotsbenders(Qx, X, benders, ncut)
    # Gets number of cuts
    nscen = length(keys(benders.cuts))
    # ncut = length(benders.cuts[nscen])

    # Adds a fictional optimal x
    xcenter = [i.first_stage.x for i in benders.solutions]

    # Computes all cuts over domain
    plotcut = [get_avg_cut(benders, i).cte .+ get_avg_cut(benders, i).ang_coef*X for i = 1:ncut]
    plotcut = hcat(plotcut...)

    # Identifies the highest cut at each x
    highest = mapslices(maximum, plotcut, dims = 2)

    # String for plot title
    if (ncut == 1) scut = "cut" else scut = "cuts" end

    # Plots
    plot(X, Qx; 
        title = "Approximated Expected Value function with $ncut $scut", xlab = "x", ylab = "Value",
        label = "True function", ylims = [-3500, -1500])
    plot!(X, plotcut; color = "orange", linestyle = :dash, alpha = 0.4, label = "")
    plot!(X, highest; color = "orange", label = "Approximated function")
end


anim = @animate for ncut in I
    plotsbenders(Qx, X, benders, ncut)
end

gif(anim, fps = 1)