########## batchcollisions
function collision_dfs(Q1::AbstractStackQtree, Q2::AbstractStackQtree, i=(levelnum(Q1), 1, 1))
    #     @show i
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
    #         @show cn,ci
    #         @show Q1[ci],Q2[ci]
        r = collision_dfs(Q1, Q2, ci)
        if r[1] > 0 return r end 
    end
    return r # no collision
end
# q1cc = [0,0,0]
# q2cc = [0,0,0]
function collision_randbfs(Q1::AbstractStackQtree, Q2::AbstractStackQtree, q=[(levelnum(Q1), 1, 1)])
#     @assert size(Q1) == size(Q2)
    if isempty(q)
        push!(q, (levelnum(Q1), 1, 1))
    end
    i = @inbounds q[1]
    while !isempty(q)
        i = popfirst!(q)
        for cn in shuffle4()
            ci = child(i, cn)
            q2 = _getindex(Q2, ci)
#             q1 = getindex(Q1, ci)
#             q1cc[q1] += 1 #result ratio [1.1, 0.04, 1.0] EMPTY > MIX > FULL
#             q2cc[q2] += 1 #result ratio [1.8, 0.07, 1.0]
            if q2 == EMPTY # assume q2 is more empty
                continue
            elseif q2 == MIX
                q1 = _getindex(Q1, ci)
                if q1 == EMPTY
                    continue
                elseif q1 == MIX
                    push!(q, ci)
                    continue
                else
                    return ci
                end
            else
                q1 = _getindex(Q1, ci)
                if q1 == EMPTY
                    continue
                end
                return ci
            end
        end
    end
    return -i[1], i[2], i[3] # no collision
end
function collision(Q1::AbstractStackQtree, Q2::AbstractStackQtree)
    l = levelnum(Q1)
    @assert l == levelnum(Q2)
    if inkernelbounds(Q1, l, 1, 1) && inkernelbounds(Q2, l, 1, 1)
        return collision_randbfs(Q1, Q2, [(l, 1, 1)])
    else
        return -l, 1, 1
    end
end
# thcc = [0 for i = 1:Threads.nthreads()]
CoItem = Pair{Tuple{Int,Int}, Index}
AbstractThreadQueue = AbstractVector{<:AbstractVector{Index}}
thread_queue() = [Vector{Tuple{Int,Int,Int}}() for i = 1:Threads.nthreads()]
# assume inkernelbounds(qtree, at) is true
function _batchcollisions_native(qtrees::AbstractVector, indpairs; 
        colist=Vector{CoItem}(),
        queue::AbstractThreadQueue=thread_queue(),
        at=(levelnum(qtrees[1]), 1, 1))
    sl = Threads.SpinLock()
    Threads.@threads for (i1, i2) in indpairs
#         thcc[Threads.threadid()] += 1
        que = queue[Threads.threadid()]
        empty!(que)
        push!(que, at)
        cp = collision_randbfs(qtrees[i1], qtrees[i2], que)
        if cp[1] >= 0
            lock(sl) do
                push!(colist, (i1, i2) => cp)
            end
        end
    end
    colist
end
function _batchcollisions_native(qtrees::AbstractVector, indpairs::Vector{CoItem}; 
    colist=Vector{CoItem}(), queue::AbstractThreadQueue=thread_queue())
    sl = Threads.SpinLock()
    Threads.@threads for ((i1, i2), at) in indpairs
#         thcc[Threads.threadid()] += 1
        que = queue[Threads.threadid()]
        empty!(que)
        push!(que, at)
        cp = collision_randbfs(qtrees[i1], qtrees[i2], que)
        if cp[1] >= 0
            lock(sl) do
                push!(colist, (i1, i2) => cp)
            end
        end
    end
    colist
end
function _batchcollisions_native(qtrees::AbstractVector, 
    inds::AbstractVector{<:Integer}=1:length(qtrees); kargs...)
    l = length(inds)
    _batchcollisions_native(qtrees, [(inds[i], inds[j]) for i in 1:l for j in l:-1:i + 1]; kargs...)
end
function _batchcollisions_native(qtrees::AbstractVector, inds::AbstractSet{<:Integer}; kargs...)
    _batchcollisions_native(qtrees, inds |> collect; kargs...)
end
function batchcollisions_native(qtrees::AbstractVector, inds=1:length(qtrees); kargs...)
    l = levelnum(qtrees[1])
    inds = [i for i in inds if inkernelbounds(@inbounds(qtrees[i][l]), 1, 1)]
    _batchcollisions_native(qtrees, inds; kargs...)
end
function locate(qt::AbstractStackQtree, ind::Index=(levelnum(qt), 1, 1))
    if qt[ind] == EMPTY
        return ind
    end
    unempty = (-1, -1, -1)
    for ci in 1:4
        c = child(ind, ci)
        if qt[c] != EMPTY
            if unempty[1] == -1 # only one empty child
                unempty = c
            else
                return ind # multiple empty child
            end
        end
    end
    return locate(qt, unempty)
end

RegionQtree{T} = QtreeNode{Pair{Index, Vector{T}}}
const NULL_NODE = RegionQtree{Any}()
const INT_NULL_NODE = RegionQtree{Int}()
nullnode(n::RegionQtree) = NULL_NODE
nullnode(n::RegionQtree{Int}) = INT_NULL_NODE
region_qtree(ind::Index, parent=NULL_NODE) = RegionQtree{Any}(ind => [], parent, [NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE])
int_region_qtree(ind::Index, parent=INT_NULL_NODE) = RegionQtree{Int}(ind => Vector{Int}(), parent, 
    [INT_NULL_NODE,INT_NULL_NODE,INT_NULL_NODE,INT_NULL_NODE])
function locate!(qt::AbstractStackQtree, regtree::QtreeNode=region_qtree((levelnum(qt), 1, 1)),
    ind::Index=(levelnum(qt), 1, 1); label=qt, newnode=region_qtree)
    if qt[ind] == EMPTY
        return regtree
    end
    locate_core!(qt, regtree, ind, label, newnode)
end
function locate_core!(qt::AbstractStackQtree, regtree::QtreeNode,
    ind::Index, label, newnode)
    if ind[1] == 1
        push!(regtree.value.second, label)
        return regtree
    end
    unempty = (-1, -1, -1)
    unemptyci = -1
    for ci in 1:4
        c = child(ind, ci)
        if _getindex(qt, c) != EMPTY
            if unemptyci == -1 # only one empty child
                unempty = c
                unemptyci = ci
            else
                push!(regtree.value.second, label)
                return regtree # multiple empty child
            end
        end
    end
    if regtree.children[unemptyci] === nullnode(regtree)
        regtree.children[unemptyci] = newnode(unempty, regtree)
    end
    locate_core!(qt, regtree.children[unemptyci], unempty, label, newnode)
end
function locate!(qts::AbstractVector, regtree::QtreeNode=int_region_qtree((levelnum(qts[1]), 1, 1))) # must have same levelnum
    for (i, qt) in enumerate(qts)
        locate!(qt, regtree, label=i, newnode=int_region_qtree)
    end
    regtree
end
function locate!(qts::AbstractVector, inds::Union{AbstractVector{Int},AbstractSet{Int}}, 
        regtree::QtreeNode=int_region_qtree((levelnum(qts[1]), 1, 1))) # must have same levelnum
    for i in inds
        locate!(qts[i], regtree, label=i, newnode=int_region_qtree)
    end
    regtree
end

function _outkernelcollision(qtrees, pos, inds, acinds, colist)
    ininds = Int[]
    for pind in acinds
        # check here because there are no bounds checking in collision_randbfs
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
function batchcollisions_region(qtrees::AbstractVector, regtree::QtreeNode; colist=Vector{CoItem}(), kargs...)
    # @assert regtree !== nullnode(regtree)
    nodequeue = [regtree]
    pairlist = Vector{CoItem}()
    while !isempty(nodequeue)
        N = popfirst!(nodequeue)
        pos, inds = N.value
        indslen = length(inds) 
        if indslen > 1
            for i in 1:indslen
                for j in indslen:-1:i+1
                    push!(pairlist, (@inbounds inds[i], @inbounds inds[j]) => pos)
                end
            end
        end
        p = N.parent
        if indslen > 0
            while p !== nullnode(regtree)
                acinds = p.value.second
                acinds = _outkernelcollision(qtrees, pos, inds, acinds, colist)
                if length(acinds) > 0
                    # append!(pairlist, (((i, pi), pos) for i in inds for pi in acinds)) #双for列表推导比Iterators.product慢，可能是长度未知append!慢
                    # append!(pairlist, [((i, pi), pos) for i in inds for pi in acinds]) 
                    append!(pairlist, ((p => pos) for p in Iterators.product(inds, acinds)))
                end
                p = p.parent
            end
        end
        for c in N.children
            if c !== nullnode(regtree)
                push!(nodequeue, c)
            end
        end
    end
    _batchcollisions_native(qtrees, pairlist; colist=colist, kargs...)
end
function batchcollisions_region(qtrees::AbstractVector; kargs...)
    regtree = locate!(qtrees)
    batchcollisions_region(qtrees, regtree; kargs...)
end
function batchcollisions_region(qtrees::AbstractVector, inds::Union{AbstractVector{Int},AbstractSet{Int}}; kargs...)
    regtree = locate!(qtrees, inds)
    batchcollisions_region(qtrees, regtree; kargs...)
end

const QTREE_COLLISION_ENABLE_TH = round(Int, 15 + 10 * log2(Threads.nthreads()))
function batchcollisions(qtrees::AbstractVector, args...; kargs...)
    if length(qtrees) > QTREE_COLLISION_ENABLE_TH
        return batchcollisions_region(qtrees, args...; kargs...)
    else
        return batchcollisions_native(qtrees, args...; kargs...)
    end
end

########## place!
function findroom_uniform(ground, q=[(levelnum(ground), 1, 1)])
    if isempty(q)
        push!(q, (levelnum(ground), 1, 1))
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
        l = max(1, levelnum(ground) - level)
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
            p1[rs + i, cs + j] = overlap(p1[rs + i, cs + j], p2[rs + i, cs + j])
        end
    end
    return p1
end

function overlap2!(tree1::ShiftedQtree, tree2::ShiftedQtree)
    overlap!(tree1[1], tree2[1])
    tree1 |> buildqtree!
end

function overlap!(tree1::ShiftedQtree, tree2::ShiftedQtree, ind::Index)
    if !(tree1[ind] == FULL || tree2[ind] == EMPTY)
        if ind[1] == 1
            tree1[ind] = FULL
        else
            for ci in 1:4
                overlap!(tree1, tree2, child(ind, ci))
            end
            qcode!(tree1, ind)
        end
    end
    tree1
end

function overlap!(tree1::ShiftedQtree, tree2::ShiftedQtree)
    @assert lastindex(tree1) == lastindex(tree2)
    @assert size(tree1[end]) == size(tree2[end]) == (1, 1)
    overlap!(tree1, tree2, (lastindex(tree1), 1, 1))
end

function overlap!(tree::ShiftedQtree, trees::AbstractVector)
    for t in trees
        overlap!(tree, t)
    end
    tree
end

"将sortedtrees依次叠加到ground上，同时修改sortedtrees的shift"
function place!(ground::ShiftedQtree, sortedtrees::AbstractVector; kargs...)
#     pos = Vector{Tuple{Int, Int, Int}}()
    ind = nothing
    for t in sortedtrees
        ind = place!(ground, t; kargs...)
        overlap!(ground, t)
        if ind === nothing
            return ind
        end
#         push!(pos, ind)
    end
    ind
#     return pos
end

function place!(ground::ShiftedQtree, qtree::ShiftedQtree; roomfinder=findroom_uniform, kargs...)
    ind = roomfinder(ground; kargs...)
    # @show ind
    if ind === nothing
        return nothing
    end
    setcenter!(qtree, getcenter(ind)) # 居中
    return ind
end

function place!(ground::ShiftedQtree, sortedtrees::AbstractVector, index::Number; kargs...)
    for i in 1:length(sortedtrees)
        if i == index
            continue
        end
        overlap!(ground, sortedtrees[i])
    end
    place!(ground, sortedtrees[index]; kargs...)
end
function place!(ground::ShiftedQtree, sortedtrees::AbstractVector, indexes; kargs...)
    for i in 1:length(sortedtrees)
        if i in indexes
            continue
        end
        overlap!(ground, sortedtrees[i])
    end
    ind = nothing
    for i in indexes
        ind = place!(ground, sortedtrees[i]; kargs...)
        if ind === nothing return ind end
        overlap!(ground, sortedtrees[i])
    end
    ind
end