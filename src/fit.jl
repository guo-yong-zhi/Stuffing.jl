module Trainer
export Momentum, train!, fit!
export trainepoch_E!, trainepoch_EM!, trainepoch_EM2!, trainepoch_EM3!, trainepoch_P!, trainepoch_P2!, trainepoch_Px!

using Random
using ..QTrees

include("traintools.jl")
mutable struct Momentum
    eta::Float64
    rho::Float64
    velocity::IdDict
end

Momentum(η, ρ=0.9) = Momentum(η, ρ, IdDict())
Momentum(;η=0.01, ρ=0.9) = Momentum(η, ρ, IdDict())

function apply(o::Momentum, x, Δ)
    η, ρ = o.eta, o.rho
    Δ = collect(Float64, Δ)
    v = get!(o.velocity, x, Δ)
    @. v = ρ * v + (1 - ρ) * Δ
    η .* v
end
function apply!(o::Momentum, x, Δ)
    Δ .= apply(o, x, Δ)
end
(opt::Momentum)(x, Δ) = apply(opt::Momentum, x, Δ)
reset!(o::Momentum, x) =  pop!(o.velocity, x)  
reset!(o, x) = nothing
Base.broadcastable(m::Momentum) = Ref(m)
@assert QTrees.EMPTY == 1 && QTrees.FULL == 2 && QTrees.MIX == 3
@inline decode2(c) = @inbounds (0, 2, 1)[c]

# const DECODETABLE = [0, 2, 1]
# near(a::Integer, b::Integer, r=1) = a-r:a+r, b-r:b+r
# near(m::AbstractMatrix, a::Integer, b::Integer, r=1) = @view m[near(a, b, r)...]
# const KERNEL = collect.(Iterators.product(-1:1,-1:1))
# gard2d(m::AbstractMatrix) = sum(KERNEL .* m)
# gard2d(t::ShiftedQtree, l, a, b) = gard2d(decode2(near(t[l],a,b)))|>Tuple

function gard2d(t::ShiftedQtree, l, a, b) # FULL is white, Positive directions are right & down 
    m = t[l]
    diag = -decode2(m[a - 1, b - 1]) + decode2(m[a + 1, b + 1])
    cdiag = -decode2(m[a - 1, b + 1]) + decode2(m[a + 1, b - 1])
    (
    + diag
    + cdiag
    - decode2(m[a - 1, b])
    + decode2(m[a + 1, b])
    ), (
    + diag
    - cdiag
    - decode2(m[a, b - 1])
    + decode2(m[a, b + 1])
    ) # (h, w)
end
# function intlog2(x::Float64) #not safe, x can't be nan or inf
#     #Float64 符号位(S)，编号63；阶码位，编号62 ~52
#     b64 = reinterpret(UInt64, x)
#     m = UInt64(0x01)<<63 #符号位mask
#     Int(1-((b64&m)>>62)), Int((b64&(~m)) >> 52 - 1023) #符号位:1-2S (1->-1、0->1)，指数位 - 1023
# end
function intlog2(x::Float64) # not safe, x>0 and x can't be nan or inf
    # Float64 符号位(S)，编号63；阶码位，编号62 ~52
    b64 = reinterpret(Int64, x)
    (b64 >> 52 - 1023) # 符号位:1-2S (1->-1、0->1)，指数位 - 1023
end

function move!(qt, ws)
    if rand() < 0.1 # 破坏周期运动
        ws = ws .+ rand(((1., -1.), (-1., 1.), (-1., -1.), (1., 1.)))
    end
    if (-1 < ws[1] < 1 && -1 < ws[2] < 1) # 避免静止
        ws = rand(((1., -1.), (-1., 1.), (-1., -1.), (1., 1.)))
    end
    wm = max(abs.(ws)...)
    # @assert wm >= 1
    u = intlog2(wm)
    # @assert u == floor(Int, log2(wm))
    shift!(qt, 1 + u, (trunc.(Int, ws) .÷ 2^u)...) # 舍尾，保留最高二进制位
end

function step!(t1, t2, collisionpoint::Tuple{Integer,Integer,Integer}, optimiser=(t, Δ) -> Δ ./ 6)
    ks1 = kernelsize(t1[1])
    ks1 = ks1[1] * ks1[2]
    ks2 = kernelsize(t2[1])
    ks2 = ks2[1] * ks2[2]
    l = collisionpoint[1]
    ll = 2^(l - 1)
#     @show collisionpoint
    ws1 = ll .* gard2d(t1, collisionpoint...)
    ws2 = ll .* gard2d(t2, collisionpoint...)
    # @assert gard2d(t1, collisionpoint...)==gard2d2(t1, collisionpoint...)
    #     @show ws1,collisionpoint,gard2d(t1, collisionpoint...)
    ws1 = optimiser(t1, ws1)
#     @show ws1
    ws2 = optimiser(t2, ws2)
    move1 = rand() < ks2 / ks1 # ks1越大移动概率越小，ks1<=ks2时必然移动（质量越大，惯性越大运动越少）
    move2 = rand() < ks1 / ks2
    if move1
        if !move2
            ws1 = ws1 .- ws2
        end
        move!(t1, ws1)
    end
    if move2
        if !move1
            ws2 = ws2 .- ws1
        end
        move!(t2, ws2)
    end
end
function step_mask!(mask, t2, collisionpoint::Tuple{Integer,Integer,Integer}, optimiser=(t, Δ) -> Δ ./ 6)
    l = collisionpoint[1]
    ll = 2^(l - 1)
    ws1 = ll .* gard2d(mask, collisionpoint...)
    ws2 = ll .* gard2d(t2, collisionpoint...)
    ws1 = optimiser(mask, ws1)
    ws2 = optimiser(t2, ws2)
    ws2 = (ws2 .- ws1) ./ 2
    move!(t2, ws2)
end

function step_ind!(qtrees, i1, i2, collisionpoint, optimiser)
#     @show i1, i2, collisionpoint
    if i1 == 1
        step_mask!(qtrees[i1], qtrees[i2], collisionpoint, optimiser)
    elseif i2 == 1
        step_mask!(qtrees[i2], qtrees[i1], collisionpoint, optimiser)
    else
        step!(qtrees[i1], qtrees[i2], collisionpoint, optimiser)
    end
end

function step_inds!(qtrees, colist::Vector{QTrees.CoItem}, optimiser)
    for ((i1, i2), cp) in shuffle!(colist)
#         @show cp
        # @assert cp[1] > 0
        step_ind!(qtrees, i1, i2, cp, optimiser)
    end
end

"element-wise trainer"
trainepoch_E!(;inputs) = Dict(:copool => Vector{QTrees.CoItem}(), :queue => QTrees.thread_queue())
trainepoch_E!(s::Symbol) = get(Dict(:patient => 10, :nepoch => 1000), s, nothing)
function trainepoch_E!(qtrees::AbstractVector{<:ShiftedQtree}; optimiser=(t, Δ) -> Δ ./ 6, 
    copool=trainepoch_E!(:copool), kargs...)
    batchcollisions(qtrees, colist=empty!(copool); kargs...)
    nc = length(copool)
    if nc == 0 return nc end
    step_inds!(qtrees, copool, optimiser)
    inds = first.(copool) |> Iterators.flatten |> Set
#     @show length(qtrees),length(inds)
    for ni in 1:length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(copool); kargs...)
        step_inds!(qtrees, copool, optimiser)
        if ni > 8length(copool) break end
    end
    nc
end

"element-wise trainer with LRU"
trainepoch_EM!(;inputs) = Dict(:copool => Vector{QTrees.CoItem}(), 
                            :queue => QTrees.thread_queue(), 
                            :memory => intlru(length(inputs)))
trainepoch_EM!(s::Symbol) = get(Dict(:patient => 10, :nepoch => 1000), s, nothing)
function trainepoch_EM!(qtrees::AbstractVector{<:ShiftedQtree}; memory, optimiser=(t, Δ) -> Δ ./ 6, 
    copool=Vector{QTrees.CoItem}(), kargs...)
    batchcollisions(qtrees, colist=empty!(copool); kargs...)
    nc = length(copool)
    if nc == 0 return nc end
    step_inds!(qtrees, copool, optimiser)
    inds = first.(copool) |> Iterators.flatten |> Set
#     @show length(inds)
    push!.(memory, inds)
    inds = take(memory, length(inds) * 2)
    for ni in 1:2length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(copool); kargs...)
        step_inds!(qtrees, copool, optimiser)
        if ni > 2length(copool) break end
        inds2 = first.(copool) |> Iterators.flatten |> Set
#         @show length(qtrees),length(inds),length(inds2)
        for ni2 in 1:2length(inds) ÷ length(inds2)
            batchcollisions(qtrees, inds2, colist=empty!(copool); kargs...)
            step_inds!(qtrees, copool, optimiser)
            if ni2 > 2length(copool) break end
        end
    end
    nc
end
"element-wise trainer with LRU(more levels)"
trainepoch_EM2!(;inputs) = trainepoch_EM!(;inputs=inputs)
trainepoch_EM2!(s::Symbol) = trainepoch_EM!(s)
function trainepoch_EM2!(qtrees::AbstractVector{<:ShiftedQtree}; memory, optimiser=(t, Δ) -> Δ ./ 6, 
    copool=Vector{QTrees.CoItem}(), kargs...)
    batchcollisions(qtrees, colist=empty!(copool); kargs...)
    nc = length(copool)
    if nc == 0 return nc end
    step_inds!(qtrees, copool, optimiser)
    inds = first.(copool) |> Iterators.flatten |> Set
#     @show length(inds)
    push!.(memory, inds)
    inds = take(memory, length(inds) * 4)
    for ni in 1:2length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(copool); kargs...)
        step_inds!(qtrees, copool, optimiser)
        if ni > 2length(copool) break end
        inds2 = first.(copool) |> Iterators.flatten |> Set
        push!.(memory, inds2)
        inds2 = take(memory, length(inds2) * 2)
        for ni2 in 1:2length(inds) ÷ length(inds2)
            batchcollisions(qtrees, inds2, colist=empty!(copool); kargs...)
            step_inds!(qtrees, copool, optimiser)
            if ni2 > 2length(copool) break end
            inds3 = first.(copool) |> Iterators.flatten |> Set
#             @show length(qtrees),length(inds),length(inds2),length(inds3)
            for ni3 in 1:2length(inds2) ÷ length(inds3)
                batchcollisions(qtrees, inds3, colist=empty!(copool); kargs...)
                step_inds!(qtrees, copool, optimiser)
                if ni3 > 2length(copool) break end
            end
        end
    end
    nc
end

"element-wise trainer with LRU(more-more levels)"
trainepoch_EM3!(;inputs) = trainepoch_EM!(;inputs=inputs)
trainepoch_EM3!(s::Symbol) = trainepoch_EM!(s)
function trainepoch_EM3!(qtrees::AbstractVector{<:ShiftedQtree}; memory, optimiser=(t, Δ) -> Δ ./ 6, 
    copool=Vector{QTrees.CoItem}(), kargs...)
    batchcollisions(qtrees, colist=empty!(copool); kargs...)
    nc = length(copool)
    if nc == 0 return nc end
    step_inds!(qtrees, copool, optimiser)
    inds = first.(copool) |> Iterators.flatten |> Set
#     @show length(inds)
    push!.(memory, inds)
    inds = take(memory, length(inds) * 8)
    for ni in 1:2length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(copool); kargs...)
        step_inds!(qtrees, copool, optimiser)
        if ni > 2length(copool) break end
        inds2 = first.(copool) |> Iterators.flatten |> Set
        push!.(memory, inds2)
        inds2 = take(memory, length(inds2) * 4)
        for ni2 in 1:2length(inds) ÷ length(inds2)
            batchcollisions(qtrees, inds2, colist=empty!(copool); kargs...)
            step_inds!(qtrees, copool, optimiser)
            if ni2 > 2length(copool) break end
            inds3 = first.(copool) |> Iterators.flatten |> Set
            push!.(memory, inds3)
            inds3 = take(memory, length(inds3) * 2)
            for ni3 in 1:2length(inds2) ÷ length(inds3)
                batchcollisions(qtrees, inds3, colist=empty!(copool); kargs...)
                step_inds!(qtrees, copool, optimiser)
                if ni3 > 2length(copool) break end
                inds4 = first.(copool) |> Iterators.flatten |> Set
#             @show length(qtrees),length(inds),length(inds2),length(inds3)
                for ni4 in 1:2length(inds3) ÷ length(inds4)
                    batchcollisions(qtrees, inds4, colist=empty!(copool); kargs...)
                    step_inds!(qtrees, copool, optimiser)
                    if ni4 > 2length(copool) break end
                end
            end
        end
    end
    nc
end

function filttrain!(qtrees, inpool, outpool, nearlevel2; optimiser, 
    queue::QTrees.AbstractThreadQueue=QTrees.thread_queue())
    nsp1 = 0
    colist = Vector{QTrees.CoItem}()
    sl1 = Threads.SpinLock()
    sl2 = Threads.SpinLock()
    Threads.@threads for (i1, i2) in inpool |> shuffle!
        que = queue[Threads.threadid()]
        cp = collision_randbfs(qtrees[i1], qtrees[i2], empty!(que))
        if cp[1] >= nearlevel2
            if outpool !== nothing
                lock(sl1) do
                    push!(outpool, (i1, i2))
                end
            end
            if cp[1] > 0
                lock(sl2) do
                    push!(colist, (i1, i2) => cp)
                end
                nsp1 += 1
            end
        end
    end
    step_inds!(qtrees, colist, optimiser)
    nsp1
end

"pairwise trainer"
trainepoch_P!(;inputs) = Dict(:copool => Vector{Tuple{Int,Int}}(),
                            :queue => QTrees.thread_queue(),
                            :nearpool => Vector{Tuple{Int,Int}}())
trainepoch_P!(s::Symbol) = get(Dict(:patient => 10, :nepoch => 100), s, nothing)
function trainepoch_P!(qtrees::AbstractVector{<:ShiftedQtree}; optimiser=(t, Δ) -> Δ ./ 6, nearlevel=-levelnum(qtrees[1]) / 2, 
    nearpool=Vector{Tuple{Int,Int}}(), copool=Vector{Tuple{Int,Int}}(), kargs...)
    nearlevel = min(-1, nearlevel)
    indpairs = [(i, j) for i in 1:length(qtrees) for j in i + 1:length(qtrees)]
    # @time 
    nsp = filttrain!(qtrees, indpairs, empty!(nearpool), nearlevel, optimiser=optimiser; kargs...)
    # @show nsp
    if nsp == 0 return 0 end 
    # @show "###",length(indpairs), length(nearpool), length(nearpool)/length(indpairs)

    # @time 
    for ni in 1:length(indpairs) ÷ length(nearpool) # the loop cost should not exceed length(indpairs)
        nsp1 = filttrain!(qtrees, nearpool, empty!(copool), 0, optimiser=optimiser; kargs...)
        # @show nsp, nsp1
        if ni > 8nsp1 break end # loop only when there are enough collisions

        for ci in 1:length(nearpool) ÷ length(copool) # the loop cost should not exceed length(nearpool)
            nsp2 = filttrain!(qtrees, copool, nothing, 0, optimiser=optimiser; kargs...)
            if ci > 4nsp2 break end # loop only when there are enough collisions
        end
        # @show length(indpairs),length(nearpool),copool
    end
    nsp
end

"pairwise trainer(more level)"
trainepoch_P2!(;inputs) = Dict(:copool => Vector{Tuple{Int,Int}}(),
                            :queue => QTrees.thread_queue(),
                            :nearpool1 => Vector{Tuple{Int,Int}}(),
                            :nearpool2 => Vector{Tuple{Int,Int}}())
trainepoch_P2!(s::Symbol) = get(Dict(:patient => 2, :nepoch => 100), s, nothing)
function trainepoch_P2!(qtrees::AbstractVector{<:ShiftedQtree}; optimiser=(t, Δ) -> Δ ./ 6, 
    nearlevel1=-levelnum(qtrees[1]) * 0.75, 
    nearlevel2=-levelnum(qtrees[1]) * 0.5, 
    nearpool1=Vector{Tuple{Int,Int}}(), 
    nearpool2=Vector{Tuple{Int,Int}}(), 
    copool=Vector{Tuple{Int,Int}}(),
    kargs...
    )
    nearlevel1 = min(-1, nearlevel1)
    nearlevel2 = min(-1, nearlevel2)

    indpairs = [(i, j) for i in 1:length(qtrees) for j in i + 1:length(qtrees)]
    nsp = filttrain!(qtrees, indpairs, empty!(nearpool1), nearlevel1, optimiser=optimiser; kargs...)
    # @show nsp
    if nsp == 0 return 0 end 
    # @show "###", length(nearpool1), length(nearpool1)/length(indpairs)

    # @time 
    for ni1 in 1:length(indpairs) ÷ length(nearpool1) # the loop cost should not exceed length(indpairs)
        nsp1 = filttrain!(qtrees, nearpool1, empty!(nearpool2), nearlevel2, optimiser=optimiser; kargs...)
        if ni1 > nsp1 break end # loop only when there are enough collisions
        # @show nsp, nsp1
        # @show "####", length(nearpool2), length(nearpool2)/length(nearpool1)

        # @time
        for ni2 in 1:length(nearpool1) ÷ length(nearpool2) # the loop cost should not exceed length(indpairs)
            nsp2 = filttrain!(qtrees, nearpool2, empty!(copool), 0, optimiser=optimiser; kargs...)
            # @show nsp2# length(copool)/length(nearpool2)
            if nsp2 == 0 || ni2 > 4 + nsp2 break end # loop only when there are enough collisions

            for ci in 1:length(nearpool2) ÷ length(copool) # the loop cost should not exceed length(nearpool)
                nsp3 = filttrain!(qtrees, copool, nothing, 0, optimiser=optimiser; kargs...)
                if ci > 4nsp3 break end # loop only when there are enough collisions
            end
            # @show length(indpairs),length(nearpool),copool
        end
    end
    nsp
end

function levelpools(qtrees, levels=[-levelnum(qtrees[1]):3:-3..., -1])
    pools = [i => Vector{Tuple{Int,Int}}() for i in levels]
#     @show typeof(pools)
    l = length(qtrees)
    for i1 in 1:l
        for i2 in i1 + 1:l
            push!(last(pools[1]), (i1, i2))
        end
    end
    pools
end
"pairwise trainer(general levels)"
trainepoch_Px!(;inputs) = Dict(:levelpools => levelpools(inputs),
                            :queue => QTrees.thread_queue())
trainepoch_Px!(s::Symbol) = get(Dict(:patient => 1, :nepoch => 10), s, nothing)
function trainepoch_Px!(qtrees::AbstractVector{<:ShiftedQtree}; 
    levelpools::AbstractVector{<:Pair{Int,<:AbstractVector{Tuple{Int,Int}}}}=levelpools(qtrees),
    optimiser=(t, Δ) -> Δ ./ 6, kargs...)
    # last_nc = typemax(Int)
    nc = 0
    if (length(levelpools) == 0) return nc end
    outpool = length(levelpools) >= 2 ? last(levelpools[2]) : nothing
    outlevel = length(levelpools) >= 2 ? first(levelpools[2]) : 0
    inpool = last(levelpools[1])
    for niter in 1:typemax(Int)
        if outpool !== nothing empty!(outpool) end
        nc = filttrain!(qtrees, inpool, outpool, outlevel, optimiser=optimiser; kargs...)
        if first(levelpools[1]) < -levelnum(qtrees[1]) + 2
            r = outpool !== nothing ? length(outpool) / length(inpool) : 1
            @info string(niter, "#"^(-first(levelpools[1])), "$(first(levelpools[1])) pool:$(length(inpool))($r) nc:$nc ")
        end
        if (nc == 0) break end
#         if (nc < last_nc) last_nc = nc else break end
        if (niter > nc) break end
        if length(levelpools) >= 2
            trainepoch_Px!(qtrees, levelpools=levelpools[2:end], optimiser=optimiser; kargs...)
        end
    end
    nc
end

function select_coinds(qtrees, copool::Vector{Tuple{Int,Int}}; on=i -> true)
    selected = Vector{Int}()
    l = length(copool)
    if l == 0
        return selected
    end
    keep = (l ./ 8 .* randexp(l)) .> 1:l # 约保留1/8
    sort!(copool, by=maximum, rev=true)
    for (i, j) in @view copool[keep]
        mij = max(i, j)
        if mij in selected || !on(mij - 1) # use index without counting mask
            continue
        end
        cp = collision_dfs(qtrees[i], qtrees[j])
        if cp[1] >= 0
            push!(selected, mij)
        end
    end
    if length(selected) == 0
        for (i, j) in @view copool[.!keep]
            mij = max(i, j)
            if !on(mij - 1)
                continue
            end
            cp = collision_dfs(qtrees[i], qtrees[j])
            if cp[1] >= 0
                push!(selected, mij)
                break
            end
        end
    end
    return selected
end
function select_coinds(qtrees, copool::Vector{QTrees.CoItem}; kargs...)
    select_coinds(qtrees, first.(copool); kargs...)
end

function reposition!(ts, copool=nothing, args...; kargs...)
    maskqt = ts[1]
    outinds = outofkernelbounds(maskqt, ts[2:end]) .+ 1
    if !isempty(outinds)
        place!(deepcopy(maskqt), ts, outinds)
        return outinds
    end
    if copool !== nothing
        selected = select_coinds(ts, copool, args...; kargs...)
        if selected !== nothing && length(selected) > 0
            place!(deepcopy(maskqt), ts, selected)
        end
        return selected
    end
    return outinds
end

function train!(ts, nepoch::Number=-1, args...; 
    trainer=trainepoch_EM2!, patient::Number=trainer(:patient), optimiser=Momentum(η=1 / 4, ρ=0.5), 
    callbackstep=1, callbackfun=x -> x, repositioning=i -> true, resource=trainer(inputs=ts), kargs...)
    reposition_flag = true
    if repositioning isa Function
        on = repositioning
    elseif repositioning isa Bool
        on = i -> repositioning
        reposition_flag = repositioning
    elseif repositioning isa AbstractFloat
        @assert 0 <= repositioning <= 1
        th = (length(ts) - 1) * (1 - repositioning)
        on = i -> i >= th
    elseif repositioning isa Int
        @assert repositioning >= 0
        on = i -> i >= repositioning
    else
        repositioning = repositioning isa AbstractSet ? repositioning : Set(repositioning)
        on = i -> i in repositioning
    end
    ep = 0
    nc = 0
    count_r = 0 #for reposition
    nc_min_r = typemax(Int) 
    count_g = 0 #for global patient
    nc_min_g = typemax(Int)
    reposition_count = 0.
    last_repositioned = nothing
    copool = nothing
    if :copool in keys(resource)
        copool = resource[:copool]
    else
        copool = resource[:levelpools][end] |> last
    end
    nepoch = nepoch >= 0 ? nepoch : trainer(:nepoch)
    @info "nepoch: $nepoch, " * (reposition_flag ? "patient: $patient" : "repositioning off")
    # curve = []
    while ep < nepoch
        nc = trainer(ts, args...; resource..., optimiser=optimiser, kargs...)
        ep += 1
        count_r += 1
        count_g += 1
        if nc < nc_min_r
            # println("   *", nc_min_r,"->",nc, " ", count_r)
            count_r = 0
            nc_min_r = nc
        end
        if nc < nc_min_g
            # println(nc_min_g,"=>",nc, " ", count_g)
            count_g = 0
            nc_min_g = nc
            # push!(curve, (ep, nc))
        end
        if nc != 0 && reposition_flag && length(ts) / 20 > length(copool) > 0 && patient > 0 && (count_r >= patient || count_r > length(copool)) # 超出耐心或少数几个碰撞
            repositioned = reposition!(ts, copool, on=on)
            @info "@epoch $ep(+$count_r), $nc($(length(copool))) collisions, reposition " * 
            (length(repositioned) > 0 ? "$repositioned to $(getshift.(ts[repositioned]))" : "nothing")
            if length(repositioned) > 0
                nc_min_r = typemax(nc_min_r)
                reset!.(optimiser, ts[repositioned])
            end
            repositioned_set = Set(repositioned)
            if last_repositioned == repositioned_set
                reposition_count += length(repositioned_set) > 0 ? 1 : 0.5
            else
                reposition_count = 0.
            end
            last_repositioned = repositioned_set
        end
        if ep % callbackstep == 0
            callbackfun(ep)
        end
        if nc == 0
            outinds = reposition!(ts)
            outlen = length(outinds)
            if outlen == 0
                break
            else
                @info "$outinds out of bounds"
                nc += outlen
            end
        end
        if reposition_count >= 10
            @info "The repositioning strategy failed after $ep epochs"
            break
        end
        if count_g > max(2, 2patient, nepoch / 50 * max(1, (length(ts) / nc_min_g)))
            @info "training early break after $ep epochs (current $nc collisions, best $nc_min_g collisions, waited $count_g epochs)"
            break
        end
    end
    # println(curve) #nc = a*exp(-b*ep)
    ep, nc
end
fit! = train!
end
