########## totalcollisions
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
    if i[1] > 1
        for cn in 0:3 # MIX
            ci = child(i, cn)
            r = collision_dfs(Q1, Q2, ci)
            if r[1] > 0 return r end 
        end
    end
    return r # no collision
end
function _collision_randbfs(Q1::AbstractStackedQTree, Q2::AbstractStackedQTree, q::AbstractVector{Index}=[(length(Q1), 1, 1)])
    #no bounds checking and assume Index[1] >= 2
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
                    if ci[1] > 1
                        push!(q, ci)
                    end
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
thread_queue() = [Vector{Tuple{Int,Int,Int}}() for i = 1:2Threads.nthreads()]
function index_chunk(l, n, ichunk) # ChunkSplitters.jl
    n_per_chunk, n_remaining = divrem(l, n)
    first = 1 + (ichunk - 1) * n_per_chunk + ifelse(ichunk <= n_remaining, ichunk - 1, n_remaining)
    last = (first - 1) + n_per_chunk + ifelse(ichunk <= n_remaining, 1, 0)
    return first:last
end
# assume inkernelbounds(qtree, at) is true
function _totalcollisions_native(qtrees::AbstractVector, copairs; 
        colist=Vector{CoItem}(),
        queue::AbstractThreadQueue=thread_queue(),
        at=(length(qtrees[1]), 1, 1))
    sl = Threads.SpinLock()
    nchunks = min(length(queue), max(1, length(copairs)÷4))
    @sync for ichunk in 1:nchunks
        que = @inbounds queue[ichunk]
        Threads.@spawn for ind in index_chunk(length(copairs), nchunks, ichunk)
            i1, i2 = copairs[ind]
            empty!(que)
            push!(que, at)
            cp = @inbounds _collision_randbfs(qtrees[i1], qtrees[i2], que)
            if cp[1] >= 0
                @Base.lock sl push!(colist, (i1, i2) => cp)
            end
        end
    end
    colist
end
function _totalcollisions_native(qtrees::AbstractVector, coitems::Vector{CoItem}; 
    colist=Vector{CoItem}(), queue::AbstractThreadQueue=thread_queue())
    sl = Threads.SpinLock()
    nchunks = min(length(queue), max(1, length(coitems)÷4))
    @sync for ichunk in 1:nchunks
        que = @inbounds queue[ichunk]
        Threads.@spawn for ind in index_chunk(length(coitems), nchunks, ichunk)
            (i1, i2), at = coitems[ind]
            empty!(que)
            push!(que, at)
            cp = @inbounds _collision_randbfs(qtrees[i1], qtrees[i2], que)
            if cp[1] >= 0
                @Base.lock sl push!(colist, (i1, i2) => cp)
            end
        end
    end
    colist
end
function _totalcollisions_native(qtrees::AbstractVector, 
    labels::AbstractVector{<:Integer}=1:length(qtrees); 
    pairlist::AbstractVector{Tuple{Int, Int}}=Vector{Tuple{Int, Int}}(), kargs...)
    l = length(labels)
    empty!(pairlist)
    append!(pairlist, (@inbounds (labels[i], labels[j]) for i in 1:l for j in l:-1:i + 1))
    _totalcollisions_native(qtrees, pairlist; kargs...)
end
function _totalcollisions_native(qtrees::AbstractVector, labels::AbstractSet{<:Integer}; kargs...)
    _totalcollisions_native(qtrees, labels |> collect; kargs...)
end
function totalcollisions_native(qtrees::AbstractVector{U8SQTree}, labels=1:length(qtrees);  
    colist=Vector{CoItem}(), kargs...)
    length(qtrees) > 1 || return colist
    l = length(@inbounds qtrees[1])
    labels = [i for i in labels if inkernelbounds(@inbounds(qtrees[i][l]), 1, 1)]
    _totalcollisions_native(qtrees, labels; colist=colist, kargs...)
end
function linked_spacial_qtree(qts)
    if !isempty(qts)
        return LinkedSpacialQTree((length(qts[1]), 1, 1), IntMap(Vector{Vector{ListNode{Int}}}(undef, length(qts))))
    else
        return LinkedSpacialQTree(IntMap(Vector{Vector{ListNode{Int}}}(undef, length(qts))))
    end
end
hash_spacial_qtree() = HashSpacialQTree()
hash_spacial_qtree(qts) = hash_spacial_qtree()

mutable struct Indices4
    i1::Index
    i2::Index
    i3::Index
	i4::Index
	len::Int
	Indices4() = (ii = new(); ii.len = 0; ii)
end
Base.getindex(ii::Indices4, i) = getfield(ii, i)
Base.setindex!(ii::Indices4, x, i) = setfield!(ii, i, x)
Base.push!(ii::Indices4, x) = ii[ii.len += 1] = x
Base.length(ii::Indices4) = ii.len
Base.iterate(ii::Indices4, i=1) = i <= length(ii) ? (ii[i], i+1) : nothing
Base.eltype(::Type{Indices4}) = Index

function locate!(qt::AbstractStackedQTree, spqtree::Union{HashSpacialQTree, LinkedSpacialQTree}, label::Int)
    l = length(qt) #l always >= 2
    # @assert kernelsize(qt[l], 1) <= 2 && kernelsize(qt[l], 2) <= 2
    while true
        l -= 1
        s1, s2 = kernelsize(@inbounds qt[l])
        if s1 > 2 || s2 > 2 || l < 2 # >= 2 for _collision_randbfs
            l += 1
            break
        end
    end
    # @assert l <= length(qt)
    # @assert l >= 2
    @inbounds mat = qt[l]
    rs, cs = getshift(mat)
    inds = Indices4()
    @inbounds mat[rs+1, cs+1] != EMPTY && push!(inds, (l, rs+1, cs+1))
    @inbounds mat[rs+1, cs+2] != EMPTY && push!(inds, (l, rs+1, cs+2))
    @inbounds mat[rs+2, cs+1] != EMPTY && push!(inds, (l, rs+2, cs+1))
    @inbounds mat[rs+2, cs+2] != EMPTY && push!(inds, (l, rs+2, cs+2))
    if length(inds) > 2 && l < length(qt)
        l = l + 1
        @inbounds mat = qt[l]
        rs, cs = getshift(mat)
        inds2 = Indices4()
        @inbounds mat[rs+1, cs+1] != EMPTY && push!(inds2, (l, rs+1, cs+1))
        @inbounds mat[rs+1, cs+2] != EMPTY && push!(inds2, (l, rs+1, cs+2))
        @inbounds mat[rs+2, cs+1] != EMPTY && push!(inds2, (l, rs+2, cs+1))
        @inbounds mat[rs+2, cs+2] != EMPTY && push!(inds2, (l, rs+2, cs+2))
        if length(inds2) < length(inds)-1
            inds = inds2
        end
    end
    push!(spqtree, inds, label)
    nothing
end
function locate!(qts::AbstractVector, spqtree=hash_spacial_qtree(qts))
    for (i, qt) in enumerate(qts)
        locate!(qt, spqtree, i)
    end
    spqtree
end
function locate!(qts::AbstractVector, labels::Union{AbstractVector{Int},AbstractSet{Int}}, spqtree=hash_spacial_qtree(qts))
    for l in labels
        locate!(qts[l], spqtree, l)
    end
    spqtree
end

function collisions_boundsfilter(qtrees, spindex, lowlabels, higlabels, itemlist, colist)
    for hlb in higlabels
        # check here because there are no bounds checking in _collision_randbfs
        collisions_boundsfilter(qtrees, spindex, lowlabels, hlb, itemlist, colist)
    end
end
function collisions_boundsfilter(qtrees, spindex, lowlabels, hlb::Int, itemlist, colist)
    if inkernelbounds(@inbounds(qtrees[hlb][spindex[1]]), spindex[2], spindex[3])
        append!(itemlist, ((llb, hlb)=>spindex for llb in lowlabels))
    elseif getdefault(@inbounds(qtrees[hlb][1])) == QTrees.FULL
        for llb in lowlabels
            if @inbounds(qtrees[llb][spindex]) != QTrees.EMPTY
                # @show (llb, hlb)=>spindex
                push!(colist, (llb, hlb) => spindex)
            end
        end
    end
end
function collisions_boundsfilter(qtrees, spindex, llb::Int, higlabels, itemlist, colist)
    collisions_boundsfilter(qtrees, spindex, (llb,), higlabels, itemlist, colist)
end
@assert collect(Iterators.product(1:2, 4:6))[1] == (1, 4)
function totalcollisions_spacial(qtrees::AbstractVector, spqtree::HashSpacialQTree; 
    colist=Vector{CoItem}(), itemlist::AbstractVector{CoItem}=Vector{CoItem}(), unique=true, kargs...)
    length(qtrees) > 1 || return colist
    nlevel = length(@inbounds qtrees[1])
    empty!(itemlist)
    for spindex in keys(spqtree)
        labels = spqtree[spindex]
        labelslen = length(labels) 
        if labelslen > 1
            for i in 1:labelslen
                for j in labelslen:-1:i+1
                    push!(itemlist, (@inbounds labels[i], @inbounds labels[j]) => spindex)
                end
            end
        end
        pspindex = spindex
        while true
            pspindex = parent(pspindex)
            (@inbounds pspindex[1] > nlevel) && break
            if haskey(spqtree, pspindex)
                plbs = spqtree[pspindex]
                collisions_boundsfilter(qtrees, spindex, labels, plbs, itemlist, colist)
            end
        end
    end
    # @show length(itemlist), length(colist)
    r = _totalcollisions_native(qtrees, itemlist; colist=colist, kargs...)
    unique ? unique!(first, sort!(r)) : r
end
function totalcollisions_spacial(qtrees::AbstractVector{U8SQTree};
    spqtree=hash_spacial_qtree(qtrees), kargs...)
    locate!(qtrees, empty!(spqtree))
    totalcollisions_spacial(qtrees, spqtree; kargs...)
end
function totalcollisions_spacial(qtrees::AbstractVector{U8SQTree}, labels::Union{AbstractVector{Int},AbstractSet{Int}}; 
    spqtree=hash_spacial_qtree(qtrees), kargs...)
    locate!(qtrees, labels, empty!(spqtree))
    totalcollisions_spacial(qtrees, spqtree; kargs...)
end

const SPACIAL_ENABLE_THRESHOLD = round(Int, 10+10log2(Threads.nthreads()))
function totalcollisions_native_kw(args...; itemlist=nothing, unique=true, spqtree=nothing, kargs...)
    totalcollisions_native(args...; kargs...)
end
totalcollisions_spacial_kw(args...; pairlist=nothing, kargs...) = totalcollisions_spacial(args...; kargs...)
function totalcollisions(args...; kargs...)
    if length(args[end]) > SPACIAL_ENABLE_THRESHOLD
        return totalcollisions_spacial_kw(args...; kargs...)
    else
        return totalcollisions_native_kw(args...; kargs...)
    end
end
function partialcollisions(qtrees::AbstractVector,
    spqtree::LinkedSpacialQTree=linked_spacial_qtree(qtrees), 
    labels::AbstractSet{Int}=Set(1:length(qtrees)); 
    colist=Vector{CoItem}(), itemlist::AbstractVector{CoItem}=Vector{CoItem}(),
    treenodestack = Vector{SpacialQTreeNode}(),
    unique=true, kargs...)
    empty!(itemlist)
    locate!(qtrees, labels, spqtree) #需要将labels中的label移动到链表首
    for label in labels
        # @show label
        for listnode in spacial_indexesof(spqtree, label)
            # 更prev的node都是move过的，在其向后遍历时会加入与当前node的pair，故不需要向前遍历
            # 但要保证更prev的node在`labels`中
            treenode = seek_treenode(listnode)
            spindex = spacial_index(treenode)
            append!(itemlist, (((label, lb) => spindex) for lb in listnode.next))
            tn = treenode
            while !isroot(tn)
                tn = tn.parent #root不是哨兵，值需要遍历
                if !isemptylabels(tn)
                    plbs = Iterators.filter(!in(labels), labelsof(tn)) #moved了的plb不加入，等候其向下遍历时加，避免重复
                    collisions_boundsfilter(qtrees, spindex, label, plbs, itemlist, colist)
                end
            end
            empty!(treenodestack)
            for c in treenode.children
                isemptychild(treenode, c) || push!(treenodestack, c)
            end
            while !isempty(treenodestack)
                tn = pop!(treenodestack)
                emptyflag = true
                if !isemptylabels(tn)
                    emptyflag = false
                    cspindex = spacial_index(tn)
                    clbs = labelsof(tn)
                    # @show cspindex clbs
                    collisions_boundsfilter(qtrees, cspindex, clbs, label, itemlist, colist)
                end
                for c in tn.children
                    if !isemptychild(tn, c) #如果isemptychild则该child无意义
                        emptyflag = false
                        push!(treenodestack, c)
                        # @show itemlist
                    end
                end
                emptyflag && remove_tree_node(spqtree, tn)
            end
        end
    end
    empty!(labels)
    # @show length(itemlist), length(colist)
    r = _totalcollisions_native(qtrees, itemlist; colist=colist, kargs...)
    unique ? unique!(first, sort!(r)) : r
end
mutable struct UpdatedSet{T} <: AbstractSet{T}
    updatelen::Int
    maxlen::Int
    set::Set{T}
end
UpdatedSet(maxlen::Int) = UpdatedSet(maxlen, maxlen, Set{Int}(1:maxlen))
UpdatedSet(labels, maxlen::Int=length(labels)) = UpdatedSet(length(labels), maxlen, Set{Int}(labels))
function Base.union!(s::UpdatedSet, c)
    s.updatelen = length(c)
    length(s.set) == s.maxlen ? s : union!(s.set, c)
end
Base.empty!(s::UpdatedSet) = empty!(s.set)
Base.length(s::UpdatedSet) = length(s.set)
Base.iterate(s::UpdatedSet, args...) = iterate(s.set, args...)
Base.in(item, s::UpdatedSet) = in(item, s.set)
Base.in(s::UpdatedSet) = in(s.set)
function totalcollisions_kw(qtrees; sptqree2=hash_spacial_qtree(qtrees), 
    spqtree=nothing, treenodestack=nothing, kargs...)
    totalcollisions(qtrees; spqtree=sptqree2, kargs...)
end
partialcollisions_kw(qtrees, spqtree, updated; sptqree2=nothing, pairlist=nothing, kargs...) = partialcollisions(qtrees, spqtree, updated; kargs...)
function dynamiccollisions(qtrees::AbstractVector,
    spqtree::LinkedSpacialQTree=linked_spacial_qtree(qtrees), 
    updated::UpdatedSet{Int}=UpdatedSet(1:length(qtrees));
    kargs...)
    if updated.updatelen / updated.maxlen > 0.6
        return totalcollisions_kw(qtrees; kargs...)
    else
        return partialcollisions_kw(qtrees, spqtree, updated; kargs...)
    end
end
struct DynamicColliders
    qtrees::Vector{U8SQTree}
    spqtree::LinkedSpacialQTree
    updated::UpdatedSet
end
DynamicColliders(qtrees::AbstractVector{U8SQTree}) = DynamicColliders(qtrees, linked_spacial_qtree(qtrees), UpdatedSet(length(qtrees)))
function dynamiccollisions(colliders::DynamicColliders; kargs...)
    r = dynamiccollisions(colliders.qtrees, colliders.spqtree, colliders.updated; kargs...)
    union!(colliders.updated, first.(r) |> Iterators.flatten)
    r
end
########## place!
function findroom_uniform(ground, q=Vector{Index}())
    if isempty(q)
        push!(q, (length(ground), 1, 1))
    end
    while !isempty(q)
        i = popfirst!(q)
#         @show i
        if i[1] == 1
            if ground[i] == EMPTY return i end
        else
            ans = nothing
            for cn in shuffle4()
                ci = child(i, cn)
                if ground[ci] == EMPTY
                    if rand() < 0.7 # 避免每次都是局部最优
                        ans = ci
                    end
                    push!(q, ci)
                elseif ground[ci] == MIX
                    push!(q, ci)
                end
            end
            if !isempty(q) && q[1][1] != i[1]
                # @assert q[1][1] == q[end][1]
                shuffle!(q)
            end
            ans !== nothing && (return ans)
        end
    end
    return nothing
end
function shufflesort!(q, ground, p)
    # @assert q[1][1] == q[end][1]
    ce = (1 + size(ground[q[1][1]], 1)) / 2
    h,w = kernelsize(ground)
    shuffle!(q)
    sort!(q, by=i -> (abs((i[2] - ce) / h)^p + (abs(i[3] - ce) / w)^p)) # 椭圆p范数
end
function findroom_gathering(ground, q=Vector{Index}(); level=5, p=2)
    if isempty(q)
        l = max(1, length(ground) - level)
        s = size(ground[l], 1)
        append!(q, ((l, i, j) for i in 1:s for j in 1:s if ground[l, i, j] != FULL))
        isempty(q) || shufflesort!(q, ground, p)
    end
    while !isempty(q)
        i = popfirst!(q)
        if i[1] == 1
            if ground[i] == EMPTY return i end
        else
            ans = nothing
            for cn in shuffle4()
                ci = child(i, cn)
                if ground[ci] == EMPTY
                    if rand() < 0.7 # 避免每次都是局部最优
                        ans = ci 
                    end
                    push!(q, ci)
                elseif ground[ci] == MIX
                    push!(q, ci)
                end
            end
            if !isempty(q) && q[1][1] != i[1]
                shufflesort!(q, ground, p)
            end
            ans !== nothing && (return ans)
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
        return MIX
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
            for ci in 0:3
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

function place!(ground::ShiftedQTree, qtree::ShiftedQTree, args...; roomfinder=findroom_uniform, kargs...)
    ind = roomfinder(ground, args...; kargs...)
    # @show ind
    if ind === nothing
        return nothing
    end
    setcenter!(qtree, getcenter(ind)) # 居中
    return ind
end

function place!(ground::ShiftedQTree, sortedtrees::AbstractVector{U8SQTree}, indexes; callback=x -> x, kargs...)
    for i in 1:length(sortedtrees)
        if i in indexes continue end
        overlap!(ground, sortedtrees[i])
    end
    ind = nothing
    Q = Vector{Index}()
    for i in indexes
        ind = place!(ground, sortedtrees[i], Q; kargs...)
        if ind === nothing return ind end
        overlap!(ground, sortedtrees[i])
        callback(i)
    end
    ind
end
function place!(ground::ShiftedQTree, sortedtrees::AbstractVector{U8SQTree}, index::Number; kargs...)
    place!(ground, sortedtrees, (index,); kargs...)
end

"将sortedtrees依次叠加到ground上，同时修改sortedtrees的shift"
function place!(ground::ShiftedQTree, sortedtrees::AbstractVector{U8SQTree}; callback=x -> x, kargs...)
    ind = nothing
    Q = Vector{Index}()
    for (i, t) in enumerate(sortedtrees)
        ind = place!(ground, t, Q; kargs...)
        if ind === nothing return ind end
        overlap!(ground, t)
        callback(i)
    end
    ind
end
