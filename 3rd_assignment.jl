# Benders decomposition for the newsvendor problem
using JuMP
using Clp
using Distributions
using Random
using Plots

include("benders.jl")

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
        ncen = length(benders.probability)
        @variable(m, z[i=1:ncen])
    end

    if length(benders.cuts) > 0
        add_cuts!(m, benders)
        if benders.avgcut
            @objective(m, Min, c*x + z)
        else
            @objective(m, Min, c*x + sum(z[i] * benders.probability[i] for i in 1:ncen))
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
        zbar = sum(value(z[i]) * benders.probability[i] for i in 1:ncen)
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