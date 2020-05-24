using JuMP
using Clp

c = 10
r=5
q=25
u=150
d = collect(60:1:u)
function news_vendor_det_eq(d,c,q,r,u)
    Ns = length(d)

    m = JuMP.Model(with_optimizer(Clp.Optimizer))
    @variable(m, 0<=x<=u)
    @variable(m, y[1:Ns]>=0)
    @variable(m, w[1:Ns]>=0)

    @constraint(m, [s=1:Ns], y[s] + w[s] <= x)
    @constraint(m, [s=1:Ns], y[s] <= d[s])

    @objective(m, Min, c*x - 1/Ns*sum(q*y[s] + r*w[s] for s in 1:Ns))
    optimize!(m)
    termination_status(m)
    return objective_value(m)
end
function news_vendor_2nd_st(d,c,q,r,u,x)
    Ns = length(d)

    m = JuMP.Model(with_optimizer(Clp.Optimizer))
    @variable(m, y[1:Ns]>=0)
    @variable(m, w[1:Ns]>=0)
    @constraint(m, [s=1:Ns], y[s] + w[s] <= x)
    @constraint(m, [s=1:Ns], y[s] <= d[s])

    @objective(m, Min, c*x - 1/Ns*sum(q*y[s] + r*w[s] for s in 1:Ns))
    optimize!(m)
    termination_status(m)
    return objective_value(m)
end
function news_vendor_expected_value(d,c,q,r,u)
    return news_vendor_det_eq(sum(d)/length(d),c,q,r,u)
end
function news_vendor_EEV(d,c,q,r,u)
    Ns = length(d)
    total = zeros(Ns)
    d_bar = sum(d)/length(d)
    # 1st stage optimal solution
    x = d_bar
    for s in 1:Ns
        total[s] = news_vendor_2nd_st(d[s], c,q,r,u, x)
    end
    return sum(total) / Ns
end
function news_vendor_perfect_info(d,c,q,r,u)
    Ns = length(d)
    total = zeros(Ns)
    for s in 1:Ns
        total[s] = news_vendor_det_eq(d[s],c,q,r,u)
    end
    return sum(total) / Ns
end

RP =  news_vendor_det_eq(d,c,q,r,u)
EEV = news_vendor_EEV(d,c,q,r,u)
PI = news_vendor_perfect_info(d,c,q,r,u)


VSS = EEV - RP
VPI = RP - PI
a = 1