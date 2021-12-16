########## batchcollisions
function collision_dfs(Q1::AbstractStackedQTree, Q2::AbstractStackedQTree, i=(length(Q1), 1, 1)) #faster than _collision_randbfs (6:7)
#     @assert size(Q1) == size(Q2)
    n1 = Q1[i]
    n2 = Q2[i]
    if n1 == EMPTY || n2 == EMPTY
        return -i[1], i[2], i[3]
    end
    if n1 == FULL || n2 == FULL
        return i
    end
    r = -i[1], i[2], i[3]
    for cn in 1:4 # MIX
        ci = child(i, cn)
        r = collision_dfs(Q1, Q2, ci)
        if r[1] > 0 return r end 
    end
    return r # no collision
end
function _collision_randbfs(Q1::AbstractStackedQTree, Q2::AbstractStackedQTree, q::AbstractVector{Index}=[(length(Q1), 1, 1)])
#     @assert size(Q1) == size(Q2)
    if isempty(q)
        push!(q, (length(Q1), 1, 1))
    end
    i = @inbounds q[1]
    while !isempty(q)
        i = popfirst!(q)
        for cn in shuffle4()
            ci = child(i, cn)
            q2 = @inbounds Q2[ci]
            if q2 == EMPTY # assume q2 is more empty
                continue
            elseif q2 == MIX
                q1 = @inbounds Q1[ci]
                if q1 == EMPTY
                    continue
                elseif q1 == MIX
                    push!(q, ci)
                    continue
                else
                    return ci
                end
            else
                q1 = @inbounds Q1[ci]
                if q1 == EMPTY
                    continue
                end
                return ci
            end
        end
    end
    return -i[1], i[2], i[3] # no collision
end
function collision(Q1::AbstractStackedQTree, Q2::AbstractStackedQTree)
    l = length(Q1)
    @assert l == length(Q2)
    if inkernelbounds(Q1, l, 1, 1) && inkernelbounds(Q2, l, 1, 1)
        return _collision_randbfs(Q1, Q2, [(l, 1, 1)])
    else
        return -l, 1, 1
    end
end
const CoItem = Pair{Tuple{Int,Int}, Index}
const AbstractThreadQueue = AbstractVector{<:AbstractVector{Index}}
thread_queue() = [Vector{Tuple{Int,Int,Int}}() for i = 1:Threads.nthreads()]
# assume inkernelbounds(qtree, at) is true
function _batchcollisions_native(qtrees::AbstractVector, indpairs; 
        colist=Vector{CoItem}(),
        queue::AbstractThreadQueue=thread_queue(),
        at=(length(qtrees[1]), 1, 1))
    sl = Threads.SpinLock()
    Threads.@threads for (i1, i2) in indpairs
        que = @inbounds queue[Threads.threadid()]
        empty!(que)
        push!(que, at)
        cp = @inbounds _collision_randbfs(qtrees[i1], qtrees[i2], que)
        if cp[1] >= 0
            @Base.lock sl push!(colist, (i1, i2) => cp)
        end
    end
    colist
end
function _batchcollisions_native(qtrees::AbstractVector, indpairs::Vector{CoItem}; 
    colist=Vector{CoItem}(), queue::AbstractThreadQueue=thread_queue())
    sl = Threads.SpinLock()
    Threads.@threads for ((i1, i2), at) in indpairs
        que = @inbounds queue[Threads.threadid()]
        empty!(que)
        push!(que, at)
        cp = @inbounds _collision_randbfs(qtrees[i1], qtrees[i2], que)
        if cp[1] >= 0
            @Base.lock sl push!(colist, (i1, i2) => cp)
        end
    end
    colist
end
function _batchcollisions_native(qtrees::AbstractVector, 
    inds::AbstractVector{<:Integer}=1:length(qtrees); kargs...)
    l = length(inds)
    _batchcollisions_native(qtrees, [@inbounds (inds[i], inds[j]) for i in 1:l for j in l:-1:i + 1]; kargs...)
end
function _batchcollisions_native(qtrees::AbstractVector, inds::AbstractSet{<:Integer}; kargs...)
    _batchcollisions_native(qtrees, inds |> collect; kargs...)
end
function batchcollisions_native(qtrees::AbstractVector{U8SQTree}, inds=1:length(qtrees); kargs...)
    l = length(qtrees[1])
    inds = [i for i in inds if inkernelbounds(@inbounds(qtrees[i][l]), 1, 1)]
    _batchcollisions_native(qtrees, inds; kargs...)
end

const RegionQTree = Dict{Index, Vector{Int}}
function locate!(qt::AbstractStackedQTree, regtree::RegionQTree, label::Int)
    l = length(qt)
    while kernelsize(@inbounds qt[l]) == (2, 2) && l >= 1
        l -= 1
    end
    l = l + 1
    @inbounds mat = qt[l]
    rs, cs = getshift(mat)
    @inbounds mat[rs+1, cs+1] != EMPTY && push!(get!(Vector{Int}, regtree, (l, rs+1, cs+1)), label)
    @inbounds mat[rs+1, cs+2] != EMPTY && push!(get!(Vector{Int}, regtree, (l, rs+1, cs+2)), label)
    @inbounds mat[rs+2, cs+1] != EMPTY && push!(get!(Vector{Int}, regtree, (l, rs+2, cs+1)), label)
    @inbounds mat[rs+2, cs+2] != EMPTY && push!(get!(Vector{Int}, regtree, (l, rs+2, cs+2)), label)
    nothing
end
function locate!(qts::AbstractVector, regtree=RegionQTree())
    for (i, qt) in enumerate(qts)
        locate!(qt, regtree, i)
    end
    regtree
end
function locate!(qts::AbstractVector, inds::Union{AbstractVector{Int},AbstractSet{Int}}, regtree=RegionQTree())
    for i in inds
        locate!(qts[i], regtree, i)
    end
    regtree
end

function outkernelcollision(qtrees, pos, inds, acinds, colist)
    ininds = Int[]
    for pind in acinds
        # check here because there are no bounds checking in _collision_randbfs
        if inkernelbounds(@inbounds(qtrees[pind][pos[1]]), pos[2], pos[3])
            push!(ininds, pind)
        elseif getdefault(@inbounds(qtrees[pind][1])) == QTrees.FULL
            for cind in inds
                if @inbounds(qtrees[cind][pos]) != QTrees.EMPTY
                    # @show (cind, pind)=>pos
                    push!(colist, (cind, pind) => pos)
                end
            end
        end
    end
    ininds
end

@assert collect(Iterators.product(1:2, 4:6))[1] == (1, 4)
function batchcollisions_region(qtrees::AbstractVector, regtree::RegionQTree; colist=Vector{CoItem}(), unique=true, kargs...)
    nlevel = length(qtrees[1])
    pairlist = Vector{CoItem}()
    for (pos, inds) in regtree
        indslen = length(inds) 
        if indslen > 1
            for i in 1:indslen
                for j in indslen:-1:i+1
                    push!(pairlist, (@inbounds inds[i], @inbounds inds[j]) => pos)
                end
            end
        end
        ppos = pos
        while true
            ppos = parent(ppos)
            (ppos[1] > nlevel) && break
            if haskey(regtree, ppos)
                pinds = regtree[ppos]
                pinds = outkernelcollision(qtrees, pos, inds, pinds, colist)
                append!(pairlist, ((p => pos) for p in Iterators.product(inds, pinds)))
            end
        end
    end
    # @show length(pairlist), length(colist)
    r = _batchcollisions_native(qtrees, pairlist; colist=colist, kargs...)
    unique ? unique!(first, sort!(r)) : r
end
function batchcollisions_region(qtrees::AbstractVector{U8SQTree}; kargs...)
    regtree = locate!(qtrees)
    batchcollisions_region(qtrees, regtree; kargs...)
end
function batchcollisions_region(qtrees::AbstractVector{U8SQTree}, inds::Union{AbstractVector{Int},AbstractSet{Int}}; kargs...)
    regtree = locate!(qtrees, inds)
    batchcollisions_region(qtrees, regtree; kargs...)
end

const QTREE_COLLISION_ENABLE_TH = round(Int, 15 + 10 * log2(Threads.nthreads()))
function batchcollisions(qtrees::AbstractVector{U8SQTree}, args...; unique=true, kargs...)
    if length(qtrees) > QTREE_COLLISION_ENABLE_TH
        return batchcollisions_region(qtrees, args...; unique=unique, kargs...)
    else
        return batchcollisions_native(qtrees, args...; kargs...)
    end
end

########## place!
function findroom_uniform(ground, q=[(length(ground), 1, 1)])
    if isempty(q)
        push!(q, (length(ground), 1, 1))
    end
    while !isempty(q)
        i = popfirst!(q)
#         @show i
        if i[1] == 1
            if ground[i] == EMPTY return i end
        else
            for cn in shuffle4()
                ci = child(i, cn)
                if ground[ci] == EMPTY
                    if rand() < 0.7 # 避免每次都是局部最优
                        return ci
                    else
                        push!(q, ci)
                    end
                elseif ground[ci] == MIX
                    push!(q, ci)
                end
            end
        end
    end
    return nothing
end
function findroom_gathering(ground, q=[]; level=5, p=2)
    if isempty(q)
        l = max(1, length(ground) - level)
        s = size(ground[l], 1)
        append!(q, ((l, i, j) for i in 1:s for j in 1:s if ground[l, i, j] != FULL))
    end
    while !isempty(q)
        # @assert q[1][1] == q[end][1]
        ce = (1 + size(ground[q[1][1]], 1)) / 2
        h,w = kernelsize(ground)
        shuffle!(q)
        sort!(q, by=i -> (abs((i[2] - ce) / h)^p + (abs(i[3] - ce) / w)^p)) # 椭圆p范数
        lq = length(q)
        for n in 1:lq
            i = popfirst!(q)
            if i[1] == 1
                if ground[i] == EMPTY return i end
            else
                for cn in shuffle4()
                    ci = child(i, cn)
                    if ground[ci] == EMPTY
                        if rand() < 0.7 # 避免每次都是局部最优
                            return ci 
                        else
                            push!(q, ci)
                        end
                    elseif ground[ci] == MIX
                        push!(q, ci)
                    end
                end
            end
        end
    end
    return nothing
end


function overlap(p1::UInt8, p2::UInt8)
    if p1 == FULL || p2 == FULL
        return FULL
    elseif p1 == EMPTY && p2 == EMPTY
        return EMPTY
    else
        error("roung code")
    end
end

overlap(p1::AbstractMatrix, p2::AbstractMatrix) = overlap.(p1, p2)

"将p2叠加到p1上"
function overlap!(p1::PaddedMat, p2::PaddedMat)
    @assert size(p1) == size(p2)
    rs, cs = getshift(p2)
    for i in 1:kernelsize(p2)[1]
        for j in 1:kernelsize(p2)[2]
            @inbounds (p1[rs + i, cs + j] = overlap(p1[rs + i, cs + j], p2[rs + i, cs + j]))
        end
    end
    return p1
end

function overlap2!(tree1::ShiftedQTree, tree2::ShiftedQTree)
    overlap!(tree1[1], tree2[1])
    tree1 |> buildqtree!
end

function _overlap!(tree1::ShiftedQTree, tree2::ShiftedQTree, ind::Index)
    if @inbounds !(tree1[ind] == FULL || tree2[ind] == EMPTY)
        if ind[1] == 1
            @inbounds tree1[ind] = FULL
        else
            for ci in 1:4
                _overlap!(tree1, tree2, child(ind, ci))
            end
            @inbounds qcode!(tree1, ind)
        end
    end
    tree1
end

function overlap!(tree1::ShiftedQTree, tree2::ShiftedQTree)
    @assert lastindex(tree1) == lastindex(tree2)
    @assert size(tree1[end]) == size(tree2[end]) == (1, 1)
    _overlap!(tree1, tree2, (lastindex(tree1), 1, 1))
end

function overlap!(tree::ShiftedQTree, trees::AbstractVector)
    for t in trees
        overlap!(tree, t)
    end
    tree
end

function place!(ground::ShiftedQTree, qtree::ShiftedQTree; roomfinder=findroom_uniform, kargs...)
    ind = roomfinder(ground; kargs...)
    # @show ind
    if ind === nothing
        return nothing
    end
    setcenter!(qtree, getcenter(ind)) # 居中
    return ind
end

function place!(ground::ShiftedQTree, sortedtrees::AbstractVector{U8SQTree}, index::Number; kargs...)
    for i in 1:length(sortedtrees)
        if i == index
            continue
        end
        overlap!(ground, sortedtrees[i])
    end
    place!(ground, sortedtrees[index]; kargs...)
end
function place!(ground::ShiftedQTree, sortedtrees::AbstractVector{U8SQTree}, indexes; callback=x -> x, kargs...)
    for i in 1:length(sortedtrees)
        if i in indexes continue end
        overlap!(ground, sortedtrees[i])
    end
    ind = nothing
    for i in indexes
        ind = place!(ground, sortedtrees[i]; kargs...)
        if ind === nothing return ind end
        overlap!(ground, sortedtrees[i])
        callback(i)
    end
    ind
end

"将sortedtrees依次叠加到ground上，同时修改sortedtrees的shift"
function place!(ground::ShiftedQTree, sortedtrees::AbstractVector{U8SQTree}; callback=x -> x, kargs...)
    ind = nothing
    for (i, t) in enumerate(sortedtrees)
        ind = place!(ground, t; kargs...)
        if ind === nothing return ind end
        overlap!(ground, t)
        callback(i)
    end
    ind
end