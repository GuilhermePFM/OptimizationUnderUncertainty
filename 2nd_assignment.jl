# SAA
using JuMP
using Clp
using Distributions
using Random
using Plots

# Build news vendor salesman problem
function news_vendor_det_eq(d,c,q,r,u)
    Ns = length(d)

    m = JuMP.Model(Clp.Optimizer)
    @variable(m, 0<=x<=u)
    @variable(m, y[1:Ns]>=0)
    @variable(m, w[1:Ns]>=0)

    @constraint(m, [s=1:Ns], y[s] + w[s] <= x)
    @constraint(m, [s=1:Ns], y[s] <= d[s])

    @objective(m, Min, c*x - 1/Ns*sum(q*y[s] + r*w[s] for s in 1:Ns))
    optimize!(m)
    termination_status(m)
    return objective_value(m), value(x)
end

# calculates SAA statistics
function SAA_stat(V, alpha)
    # V: set of SAA solutions for M scenarios sampling for the problem

    M = length(V)
    # estimator for LB
    avg = sum(V)/M

    # variance
    var = sum( (s-avg)^2 / (M - 1) for s in V )

    # standard deviation
    std = sqrt(var)

    # confidence interval
    Z = Normal()
    conf_lb = avg - quantile(Z, alpha) * std / sqrt(M)
    conf_ub = avg + quantile(Z, alpha) * std / sqrt(M)

    return avg, var, std, (conf_lb, conf_ub)
end

# build objective cost for SAA UB calculation
y(x,d) = min(x,d)
w(x,d) = max(0, x - d)
function fobj(d, c, q, r, u, x)
    # optimal policy
    Ns = length(d)
    return c*x - 1/Ns*sum(q*y(x,s) + r*w(x,s) for s in d)
end

# 1) Using Sample Average Approximations, compute and plot the confidence interval for the lower bound for a fixed number of replications M=10 and a varying number of in-sample scenarios N \in \{50,100, \ldots,1000\}
function problem_1()
    M = 10
    N = collect(50:50:1000)
    c = 10
    r=5
    q=25
    u=150
    uni_dist = Uniform(50,150)
    Random.seed!(123)

    alpha = 0.95
    int_LB=[]
    int_UB=[]
    for Ns in N
        # Ns sample for each replica
        S = [rand(uni_dist, Ns) for i in 1:M]

        # for each replica, calculates the deterministic eq
        V = []
        X = []
        for d in S
            v, x = news_vendor_det_eq(d,c,q,r,u)
            push!(V,v)
            push!(X,x)
        end

        # the lower bound is calulated with the results from the replicas:
        LB_avg, LB_var, LB_std, LB_conf = SAA_stat(V, alpha)
        push!(int_LB, LB_conf[1])
        push!(int_UB, LB_conf[2])
    end
    x=N
    plot(x, int_LB, title = "Lower bound confidence interval", label = "LB lower")
    plot!(x, int_UB, label = "LB upper")
    plot!(x, [-1312.5 for i in x], label = "LB upper")
end

# 2) Assuming N=1000, for each of the M=10 solution candidates, compute the confidence interval for the upper bound for the number of out-of-sample scenarios K = 1000
function problem_2()
    M = 10
    Ns = 1000
    K = 1000
    c = 10
    r=5
    q=25
    u=150
    uni_dist = Uniform(50,150)
    Random.seed!(123)

    alpha = 0.95
    # Ns sample for each replica
    S = [rand(uni_dist, Ns) for i in 1:M]

    # for each replica, calculates the deterministic eq
    V = []
    X = []
    for d in S
        v, x = news_vendor_det_eq(d,c,q,r,u)
        push!(V,v)
        push!(X,x)
    end

    # the lower bound is calulated with the results from the replicas:
    LB_avg, LB_var, LB_std, LB_conf = SAA_stat(V, alpha)

    # the upper bound is calculated with optimal x of each replica M in each scenario K
    Ks = rand(uni_dist, K)
    int_LB=[]
    int_UB=[]
    xs = sort(X)
    for x in xs
        W = [fobj(d, c, q, r, u, x) for d in Ks]
        UB_avg, UB_var, UB_std, UB_conf = SAA_stat(W, alpha)
        push!(int_LB, UB_conf[1])
        push!(int_UB, UB_conf[2])
    end
    plot(xs, int_LB, title = "Upper bound confidence interval for each solution", label = "UB lower", seriestype = :scatter)
    plot!(xs, int_UB, label = "UB upper", seriestype = :scatter)
    xlabel!("X")
    ylabel!("Cost")
end

# 3) Assuming N=1000, for each of the 10 solution candidates, assuming M = 10 and K = 1000, compute the confidence interval for the gap for K = 1000
# 4) Choose a candidate solution and justify your choice.
# and
# 5) Now, for the chosen solution, compute and plot the confidence interval for the UB for differing K \in \{100, 200, ..., 1000\}.
function problem_3()
    M = 10
    Ns = 1000
    K = 1000
    c = 10
    r=5
    q=25
    u=150
    uni_dist = Uniform(50,150)
    Random.seed!(123)

    alpha = 0.95
    # Ns sample for each replica
    S = [rand(uni_dist, Ns) for i in 1:M]

    # for each replica, calculates the deterministic eq
    V = []
    X = []
    for d in S
        v, x = news_vendor_det_eq(d,c,q,r,u)
        push!(V,v)
        push!(X,x)
    end

    # the lower bound is calulated with the results from the replicas:
    LB_avg, LB_var, LB_std, LB_conf = SAA_stat(V, alpha)

    # the upper bound is calculated with optimal x of each replica M in each scenario K
    Ks = rand(uni_dist, K)
    Z = Normal()
    int_LB=[]
    int_UB=[]
    GAP_avg_l=[]
    UB_avg_l=[]
    xs = sort(X)
    for x in xs
        W = [fobj(d, c, q, r, u, x) for d in Ks]
        UB_avg, UB_var, UB_std, UB_conf = SAA_stat(W, alpha)

        # GAP calculations
        GAP_avg = UB_avg - LB_avg
        GAP_var = UB_var / K + LB_var / M

        std = sqrt(GAP_var)
        GAP_int_LB = GAP_avg - quantile(Z, alpha) * std
        GAP_int_UB = GAP_avg + quantile(Z, alpha) * std
        
        # push
        push!(GAP_avg_l, GAP_avg)
        push!(int_LB, GAP_int_LB)
        push!(int_UB, GAP_int_UB)
        push!(UB_avg_l, UB_avg)
    end
    plot(xs, int_LB, title = "GAP confidence interval for each solution", label = "GAP lower", seriestype = :scatter)
    plot!(xs, int_UB, label = "GAP upper", seriestype = :scatter)
    xlabel!("X")
    ylabel!("GAP")

    # 4) Choose a candidate solution and justify your choice.
    # ub_sol, i = findmin(int_UB.-int_LB)
    ub_sol, i = findmin(UB_avg_l)
    x_sol = xs[i]
    int_LB_sol = int_LB[i]
    int_UB_sol = int_UB[i]

    # 5) Now, for the chosen solution, compute and plot the confidence interval for the UB for differing K \in \{100, 200, ..., 1000\}.
    int_LB= []
    int_UB= []
    ks = 100:100:K
    for k in ks
        Ks = rand(uni_dist, k)
        W = [fobj(d, c, q, r, u, x_sol) for d in Ks]
        UB_avg, UB_var, UB_std, UB_conf = SAA_stat(W, alpha)

        push!(int_LB, UB_conf[1])
        push!(int_UB, UB_conf[2])
    end
    plot(ks, int_LB, title = "UB confidence interval for solution x = $x_sol", label = "UB lower")
    plot!(ks, int_UB, label = "UB upper")
    plot!(ks, [-1312.5 for i in ks], label = "LB upper")
    xlabel!("K")
    ylabel!("UB bounds")
end
