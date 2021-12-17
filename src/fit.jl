module Trainer
export Momentum, SGD, train!, fit!
export trainepoch_E!, trainepoch_EM!, trainepoch_EM2!, trainepoch_EM3!, trainepoch_P!, trainepoch_P2!, trainepoch_Px!

using Random
using ..QTrees

include("fit_helper.jl")

function move!(qt, delta)
    if rand() < 0.1 # 破坏周期运动
        delta = delta .+ rand(((1., -1.), (-1., 1.), (-1., -1.), (1., 1.)))
    end
    if (-1 < delta[1] < 1 && -1 < delta[2] < 1) # 避免静止
        delta = rand(((1., -1.), (-1., 1.), (-1., -1.), (1., 1.)))
    end
    delta_m = max(abs.(delta)...)
    # @assert delta_m >= 1
    u = intlog2(delta_m)
    # @assert u == floor(Int, log2(delta_m))
    shift!(qt, 1 + u, (trunc.(Int, delta) .÷ 2^u)...) # 舍尾，保留最高二进制位
end

function step!(t1, t2, collisionpoint::Tuple{Integer,Integer,Integer}, optimiser=(t, Δ) -> Δ ./ 6)
    ks1 = kernelsize(t1[1])
    ks1 = (ks1[1] + ks1[2]) ^ 3 #1阶抵消碰撞发生概率，2阶是惯性
    ks2 = kernelsize(t2[1])
    ks2 = (ks2[1] + ks2[2]) ^ 3
    l = collisionpoint[1]
    ll = 2^(l - 1)
    # @show collisionpoint
    delta1 = ll .* gard2d(t1, collisionpoint...)
    delta2 = ll .* gard2d(t2, collisionpoint...)
    # @assert gard2d(t1, collisionpoint...)==gard2d2(t1, collisionpoint...)
    # @show delta1,collisionpoint,gard2d(t1, collisionpoint...)
    delta1 = optimiser(t1, delta1)
    # @show delta1
    delta2 = optimiser(t2, delta2)
    move1 = rand() < ks2 / ks1 # ks1越大移动概率越小，ks1<=ks2时必然移动（质量越大，惯性越大运动越少）
    move2 = rand() < ks1 / ks2
    if move1
        if !move2
            delta1 = 2 .* delta1
        end
        move!(t1, delta1)
    end
    if move2
        if !move1
            delta2 = 2 .* delta2
        end
        move!(t2, delta2)
    end
end
function step_mask!(mask, t2, collisionpoint::Tuple{Integer,Integer,Integer}, optimiser=(t, Δ) -> Δ ./ 6)
    l = collisionpoint[1]
    ll = 2^(l - 1)
    delta1 = ll .* gard2d(mask, collisionpoint...)
    delta2 = ll .* gard2d(t2, collisionpoint...)
    delta1 = optimiser(mask, delta1)
    delta2 = optimiser(t2, delta2)
    delta2 = (delta2 .- delta1) ./ 2
    move!(t2, delta2)
end

function step_ind!(qtrees, i1, i2, collisionpoint, optimiser)
    # @show i1, i2, collisionpoint
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
        # @show cp
        # @assert cp[1] > 0
        step_ind!(qtrees, i1, i2, cp, optimiser)
    end
end

"element-wise trainer"
trainepoch_E!(;inputs) = Dict(:colist => Vector{QTrees.CoItem}(), :queue => QTrees.thread_queue())
trainepoch_E!(s::Symbol) = get(Dict(:patient => 10, :nepoch => 1000), s, nothing)
function trainepoch_E!(qtrees::AbstractVector{<:ShiftedQTree}; optimiser=(t, Δ) -> Δ ./ 6, 
    colist=trainepoch_E!(:colist), kargs...)
    batchcollisions(qtrees, colist=empty!(colist); kargs...)
    nc = length(colist)
    if nc == 0 return nc end
    step_inds!(qtrees, colist, optimiser)
    inds = first.(colist) |> Iterators.flatten |> Set
    # @show length(qtrees),length(inds)
    for ni in 1:length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(colist); kargs...)
        step_inds!(qtrees, colist, optimiser)
        if ni > 8length(colist) break end
    end
    nc
end

"element-wise trainer with LRU"
trainepoch_EM!(;inputs) = Dict(:colist => Vector{QTrees.CoItem}(), 
                            :queue => QTrees.thread_queue(), 
                            :memory => intlru(length(inputs)))
trainepoch_EM!(s::Symbol) = get(Dict(:patient => 10, :nepoch => 1000), s, nothing)
function trainepoch_EM!(qtrees::AbstractVector{<:ShiftedQTree}; memory, optimiser=(t, Δ) -> Δ ./ 6, 
    colist=Vector{QTrees.CoItem}(), kargs...)
    batchcollisions(qtrees, colist=empty!(colist); kargs...)
    nc = length(colist)
    if nc == 0 return nc end
    step_inds!(qtrees, colist, optimiser)
    inds = first.(colist) |> Iterators.flatten |> Set
    # @show length(inds)
    push!.(memory, inds)
    inds = take(memory, length(inds) * 2)
    for ni in 1:2length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(colist); kargs...)
        step_inds!(qtrees, colist, optimiser)
        if ni > 2length(colist) break end
        inds2 = first.(colist) |> Iterators.flatten |> Set
        # @show length(qtrees),length(inds),length(inds2)
        for ni2 in 1:2length(inds) ÷ length(inds2)
            batchcollisions(qtrees, inds2, colist=empty!(colist); kargs...)
            step_inds!(qtrees, colist, optimiser)
            if ni2 > 2length(colist) break end
        end
    end
    nc
end
"element-wise trainer with LRU(more levels)"
trainepoch_EM2!(;inputs) = trainepoch_EM!(;inputs=inputs)
trainepoch_EM2!(s::Symbol) = trainepoch_EM!(s)
function trainepoch_EM2!(qtrees::AbstractVector{<:ShiftedQTree}; memory, optimiser=(t, Δ) -> Δ ./ 6, 
    colist=Vector{QTrees.CoItem}(), kargs...)
    batchcollisions(qtrees, colist=empty!(colist); kargs...)
    nc = length(colist)
    if nc == 0 return nc end
    step_inds!(qtrees, colist, optimiser)
    inds = first.(colist) |> Iterators.flatten |> Set
    # @show length(inds)
    push!.(memory, inds)
    inds = take(memory, length(inds) * 4)
    for ni in 1:2length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(colist); kargs...)
        step_inds!(qtrees, colist, optimiser)
        if ni > 2length(colist) break end
        inds2 = first.(colist) |> Iterators.flatten |> Set
        push!.(memory, inds2)
        inds2 = take(memory, length(inds2) * 2)
        for ni2 in 1:2length(inds) ÷ length(inds2)
            batchcollisions(qtrees, inds2, colist=empty!(colist); kargs...)
            step_inds!(qtrees, colist, optimiser)
            if ni2 > 2length(colist) break end
            inds3 = first.(colist) |> Iterators.flatten |> Set
            # @show length(qtrees),length(inds),length(inds2),length(inds3)
            for ni3 in 1:2length(inds2) ÷ length(inds3)
                batchcollisions(qtrees, inds3, colist=empty!(colist); kargs...)
                step_inds!(qtrees, colist, optimiser)
                if ni3 > 2length(colist) break end
            end
        end
    end
    nc
end

"element-wise trainer with LRU(more-more levels)"
trainepoch_EM3!(;inputs) = trainepoch_EM!(;inputs=inputs)
trainepoch_EM3!(s::Symbol) = trainepoch_EM!(s)
function trainepoch_EM3!(qtrees::AbstractVector{<:ShiftedQTree}; memory, optimiser=(t, Δ) -> Δ ./ 6, 
    colist=Vector{QTrees.CoItem}(), kargs...)
    batchcollisions(qtrees, colist=empty!(colist); kargs...)
    nc = length(colist)
    if nc == 0 return nc end
    step_inds!(qtrees, colist, optimiser)
    inds = first.(colist) |> Iterators.flatten |> Set
    # @show length(inds)
    push!.(memory, inds)
    inds = take(memory, length(inds) * 8)
    for ni in 1:2length(qtrees) ÷ length(inds)
        batchcollisions(qtrees, inds, colist=empty!(colist); kargs...)
        step_inds!(qtrees, colist, optimiser)
        if ni > 2length(colist) break end
        inds2 = first.(colist) |> Iterators.flatten |> Set
        push!.(memory, inds2)
        inds2 = take(memory, length(inds2) * 4)
        for ni2 in 1:2length(inds) ÷ length(inds2)
            batchcollisions(qtrees, inds2, colist=empty!(colist); kargs...)
            step_inds!(qtrees, colist, optimiser)
            if ni2 > 2length(colist) break end
            inds3 = first.(colist) |> Iterators.flatten |> Set
            push!.(memory, inds3)
            inds3 = take(memory, length(inds3) * 2)
            for ni3 in 1:2length(inds2) ÷ length(inds3)
                batchcollisions(qtrees, inds3, colist=empty!(colist); kargs...)
                step_inds!(qtrees, colist, optimiser)
                if ni3 > 2length(colist) break end
                inds4 = first.(colist) |> Iterators.flatten |> Set
            # @show length(qtrees),length(inds),length(inds2),length(inds3)
                for ni4 in 1:2length(inds3) ÷ length(inds4)
                    batchcollisions(qtrees, inds4, colist=empty!(colist); kargs...)
                    step_inds!(qtrees, colist, optimiser)
                    if ni4 > 2length(colist) break end
                end
            end
        end
    end
    nc
end

function filttrain!(qtrees, inpool, outpool, nearlevel2; optimiser, 
    queue::QTrees.AbstractThreadQueue=QTrees.thread_queue(), unique=true)
    nc1 = 0
    colist = Vector{QTrees.CoItem}()
    sl1 = Threads.SpinLock()
    sl2 = Threads.SpinLock()
    Threads.@threads for (i1, i2) in inpool |> shuffle!
        que = queue[Threads.threadid()]
        cp = QTrees._collision_randbfs(qtrees[i1], qtrees[i2], empty!(que))
        if cp[1] >= nearlevel2
            if outpool !== nothing
                @Base.lock sl1 push!(outpool, (i1, i2))
            end
            if cp[1] > 0
                @Base.lock sl2 push!(colist, (i1, i2) => cp)
                nc1 += 1
            end
        end
    end
    step_inds!(qtrees, colist, optimiser)
    nc1
end

"pairwise trainer"
trainepoch_P!(;inputs) = Dict(:colist => Vector{Tuple{Int,Int}}(),
                            :queue => QTrees.thread_queue(),
                            :nearlist => Vector{Tuple{Int,Int}}())
trainepoch_P!(s::Symbol) = get(Dict(:patient => 10, :nepoch => 100), s, nothing)
function trainepoch_P!(qtrees::AbstractVector{<:ShiftedQTree}; optimiser=(t, Δ) -> Δ ./ 6, nearlevel=-length(qtrees[1]) / 2, 
    nearlist=Vector{Tuple{Int,Int}}(), colist=Vector{Tuple{Int,Int}}(), kargs...)
    nearlevel = min(-1, nearlevel)
    indpairs = [(i, j) for i in 1:length(qtrees) for j in i+1:length(qtrees)]
    nc = filttrain!(qtrees, indpairs, empty!(nearlist), nearlevel, optimiser=optimiser; kargs...)
    # @show nc
    if nc == 0 return 0 end 
    # @show "###",length(indpairs), length(nearlist), length(nearlist)/length(indpairs)

    for ni in 1:length(indpairs) ÷ length(nearlist) # the loop cost should not exceed length(indpairs)
        nc1 = filttrain!(qtrees, nearlist, empty!(colist), 0, optimiser=optimiser; kargs...)
        # @show nc, nc1
        if ni > 8nc1 break end # loop only when there are enough collisions

        for ci in 1:length(nearlist) ÷ length(colist) # the loop cost should not exceed length(nearlist)
            nc2 = filttrain!(qtrees, colist, nothing, 0, optimiser=optimiser; kargs...)
            if ci > 4nc2 break end # loop only when there are enough collisions
        end
        # @show length(indpairs),length(nearlist),colist
    end
    nc
end

"pairwise trainer(more level)"
trainepoch_P2!(;inputs) = Dict(:colist => Vector{Tuple{Int,Int}}(),
                            :queue => QTrees.thread_queue(),
                            :nearlist1 => Vector{Tuple{Int,Int}}(),
                            :nearlist2 => Vector{Tuple{Int,Int}}())
trainepoch_P2!(s::Symbol) = get(Dict(:patient => 2, :nepoch => 100), s, nothing)
function trainepoch_P2!(qtrees::AbstractVector{<:ShiftedQTree}; optimiser=(t, Δ) -> Δ ./ 6, 
    nearlevel1=-length(qtrees[1]) * 0.75, 
    nearlevel2=-length(qtrees[1]) * 0.5, 
    nearlist1=Vector{Tuple{Int,Int}}(), 
    nearlist2=Vector{Tuple{Int,Int}}(), 
    colist=Vector{Tuple{Int,Int}}(),
    kargs...
    )
    nearlevel1 = min(-1, nearlevel1)
    nearlevel2 = min(-1, nearlevel2)

    indpairs = [(i, j) for i in 1:length(qtrees) for j in i+1:length(qtrees)]
    nc = filttrain!(qtrees, indpairs, empty!(nearlist1), nearlevel1, optimiser=optimiser; kargs...)
    # @show nc
    if nc == 0 return 0 end 
    # @show "###", length(nearlist1), length(nearlist1)/length(indpairs)

    for ni1 in 1:length(indpairs) ÷ length(nearlist1) # the loop cost should not exceed length(indpairs)
        nc1 = filttrain!(qtrees, nearlist1, empty!(nearlist2), nearlevel2, optimiser=optimiser; kargs...)
        if ni1 > nc1 break end # loop only when there are enough collisions
        # @show nc, nc1
        # @show "####", length(nearlist2), length(nearlist2)/length(nearlist1)

        for ni2 in 1:length(nearlist1) ÷ length(nearlist2) # the loop cost should not exceed length(indpairs)
            nc2 = filttrain!(qtrees, nearlist2, empty!(colist), 0, optimiser=optimiser; kargs...)
            # @show nc2# length(colist)/length(nearlist2)
            if nc2 == 0 || ni2 > 4 + nc2 break end # loop only when there are enough collisions

            for ci in 1:length(nearlist2) ÷ length(colist) # the loop cost should not exceed length(nearlist)
                nc3 = filttrain!(qtrees, colist, nothing, 0, optimiser=optimiser; kargs...)
                if ci > 4nc3 break end # loop only when there are enough collisions
            end
            # @show length(indpairs),length(nearlist),colist
        end
    end
    nc
end

function levelpools(qtrees, levels=[-length(qtrees[1]):3:-3..., -1])
    pools = [i => Vector{Tuple{Int,Int}}() for i in levels]
    l = length(qtrees)
    for i1 in 1:l
        for i2 in i1+1:l
            push!(last(pools[1]), (i1, i2))
        end
    end
    pools
end
"pairwise trainer(general levels)"
trainepoch_Px!(;inputs) = Dict(:levelpools => levelpools(inputs),
                            :queue => QTrees.thread_queue())
trainepoch_Px!(s::Symbol) = get(Dict(:patient => 1, :nepoch => 10), s, nothing)
function trainepoch_Px!(qtrees::AbstractVector{<:ShiftedQTree}; 
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
        if first(levelpools[1]) < -length(qtrees[1]) + 2
            r = outpool !== nothing ? length(outpool) / length(inpool) : 1
            @info string(niter, "#"^(-first(levelpools[1])), "$(first(levelpools[1])) pool:$(length(inpool))($r) nc:$nc ")
        end
        if (nc == 0) break end
        # if (nc < last_nc) last_nc = nc else break end
        if (niter > nc) break end
        if length(levelpools) >= 2
            trainepoch_Px!(qtrees, levelpools=levelpools[2:end], optimiser=optimiser; kargs...)
        end
    end
    nc
end

function select_coinds(qtrees, colist::Vector{Tuple{Int,Int}}; from=i->true)
    selected = Vector{Int}()
    l = length(colist)
    if l == 0
        return selected
    end
    keep = (l ./ 8 .* randexp(l)) .> 1:l # 约保留1/8
    sort!(colist, by=maximum, rev=true)
    for (i, j) in @view colist[keep]
        mij = max(i, j)
        if mij in selected || !from(mij - 1) # use index without counting mask
            continue
        end
        cp = collision_dfs(qtrees[i], qtrees[j])
        if cp[1] >= 0
            push!(selected, mij)
        end
    end
    if length(selected) == 0
        for (i, j) in @view colist[.!keep]
            mij = max(i, j)
            if !from(mij - 1)
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
function select_coinds(qtrees, colist::Vector{QTrees.CoItem}; kargs...)
    select_coinds(qtrees, first.(colist); kargs...)
end

function reposition!(ts, colist=nothing, args...; kargs...)
    maskqt = ts[1]
    outinds = outofkernelbounds(maskqt, ts[2:end]) .+ 1
    if !isempty(outinds)
        place!(deepcopy(maskqt), ts, outinds)
        return outinds
    end
    if colist !== nothing
        selected = select_coinds(ts, colist, args...; kargs...)
        if selected !== nothing && length(selected) > 0
            place!(deepcopy(maskqt), ts, selected)
        end
        return selected
    end
    return outinds
end

function train!(ts, nepoch::Number=-1, args...; 
    trainer=trainepoch_EM2!, patient::Number=trainer(:patient), 
    optimiser=Momentum(η=1/4), scheduler=lr->lr*√0.5,
    callback=x -> x, reposition=i -> true, resource=trainer(inputs=ts), kargs...)
    reposition_flag = true
    if reposition isa Function
        from = reposition
    elseif reposition isa Bool
        from = i -> reposition
        reposition_flag = reposition
    elseif reposition isa AbstractFloat
        @assert 0 <= reposition <= 1
        th = (length(ts) - 1) * (1 - reposition)
        from = i -> i >= th
    elseif reposition isa Int
        @assert reposition >= 0
        from = i -> i >= reposition
    else
        reposition isa AbstractSet || (reposition = Set(reposition))
        from = i -> i in reposition
    end
    ep = 0
    nc = 0
    indi_r = MonotoneIndicator{Int}() #for reposition
    indi_g = MonotoneIndicator{Int}() #for global patient
    indi_s = MonotoneIndicator{Int}() #for lr scheduler
    eta_list = []
    reposition_count = 0.
    last_repositioned = nothing
    colist = nothing
    if :colist in keys(resource)
        colist = resource[:colist]
    else
        colist = resource[:levelpools][end] |> last
    end
    nepoch >= 0 || (nepoch = trainer(:nepoch))
    @info "nepoch: $nepoch, " * (reposition_flag ? "patient: $patient" : "reposition off")
    while ep < nepoch
        nc = trainer(ts, args...; resource..., optimiser=optimiser, unique=false, kargs...)
        ep += 1
        update!(indi_r, nc)
        update!(indi_g, nc)
        update!(indi_s, nc)
        if nc != 0 && reposition_flag && length(ts) / 20 > length(colist) > 0 && patient > 0 && (indi_r.age >= patient || indi_r.age > length(colist)) # 超出耐心或少数几个碰撞
            repositioned = reposition!(ts, colist, from=from)
            @info "@epoch $ep(+$(indi_r.age)), $nc($(length(colist))) collisions, reposition " * 
            (length(repositioned) > 0 ? "$repositioned to $(getshift.(ts[repositioned]))" : "nothing")
            if length(repositioned) > 0
                reset!(indi_r)
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
        callback(ep)
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
        if indi_s.age > max(1, patient, nepoch / 50)
            if isempty(eta_list) || indi_s.min < eta_list[end][2] || (indi_s.min == eta_list[end][2] && rand()>0.5)
                _eta = eta(optimiser)
                push!(eta_list, (_eta, indi_s.min))
                eta!(optimiser, scheduler(_eta))
                @info "@epoch $ep(+$(indi_s.age)) η -> $(round(eta(optimiser), digits=3)) (current $nc collisions, best $(indi_s.min) collisions)"
            else
                last_eta, last_nc = pop!(eta_list)
                eta!(optimiser, last_eta)
                @info "@epoch $ep(+$(indi_s.age)) η <- $(round(eta(optimiser), digits=3)) (collisions $(indi_s.min) ≥ $(last_nc))"
            end
            reset!(indi_s)
        end
        if indi_g.age > max(2, 2patient, nepoch / 50 * max(1, (length(ts) / indi_g.min)))
            @info "training early break after $ep(+$(indi_g.age)) epochs (current $nc collisions, best $(indi_g.min) collisions)"
            break
        end
    end
    ep, nc
end
fit! = train!
end
