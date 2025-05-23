module QTrees
export AbstractStackedQTree, StackedQTree, ShiftedQTree, buildqtree!,
    shift!, setrshift!, setcshift!, setshift!, getshift, getcenter, setcenter!,
    collision, collision_dfs, totalcollisions, partialcollisions, dynamiccollisions,
    locate!, linked_spacial_qtree, hash_spacial_qtree, DynamicColliders,
    findroom_uniform, findroom_gathering, outofbounds, outofkernelbounds, 
    kernelsize, place!, overlap, overlap!, decode, charimage

using Random
using ..CommonDatatypes

const PERM4 = ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), 
(1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0), 
(2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0), 
(3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0))
@assert length(PERM4) == 24
@inline shuffle4() = @inbounds PERM4[rand(1:24)]
const Index = Tuple{Int, Int, Int}
@inline function child(ind::Index, n::Int) # n: 0, 1, 2, 3
    # @assert 0 <= n <= 3
    @inbounds (ind[1] - 1, 2ind[2] - n & 1, 2ind[3] - n >> 1)
end
@inline parent(ind::Index) = @inbounds (ind[1] + 1, (ind[2] + 1) ÷ 2, (ind[3] + 1) ÷ 2)
indexcenter(l::Integer, a::Integer, b::Integer) = l == 1 ? (a, b) : (2^(l - 1) * (a - 1) + 2^(l - 2), 2^(l - 1) * (b - 1) + 2^(l - 2))
indexcenter(ind) = indexcenter(ind...)
function childnumber(ancestor::Index, descendant::Index) #assume the ancestor-descendant relationship exists
    o2, o3 = indexcenter(ancestor[1] - descendant[1] + 1, ancestor[2], ancestor[3])
    ((descendant[3] <= o3) << 1) | (descendant[2] <= o2) # 0, 1, 2, 3
end
function childnumber(ind::Index)
    ((ind[3] & 1) << 1) | (ind[2] & 1) # 0, 1, 2, 3
end
function indexrange(l::Integer, a::Integer, b::Integer)
    r = 2^(l - 1)
    r * (a - 1) + 1:r * a, r * (b - 1) + 1:r * b
end
indexrange(ind) = indexrange(ind...)
function inrange(a::Index, b::Index)
    a1, a2, a3 = a
    b1, b2, b3 = b
    if a1 > b1
        r2, r3 = indexrange(a1-b1+1, a2, a3)
        return b2 in r2 && b3 in r3
    elseif a1 == b1
        return a2 == b2 && a3 == b3
    else
        error("$a1 < $b1")
    end
end
Base.@propagate_inbounds function qcode(Q, i)
    @inbounds (Q[child(i, 0)] | Q[child(i, 1)] | Q[child(i, 2)] | Q[child(i, 3)])
end
Base.@propagate_inbounds qcode!(Q, i) = @inbounds Q[i] = qcode(Q, i)
decode(c) = (0., 1., 0.5)[c]

const FULL = 0x02; const EMPTY = 0x01; const MIX = 0x03

abstract type AbstractStackedQTree end
Base.@propagate_inbounds function Base.getindex(t::AbstractStackedQTree, l::Integer) end
Base.@propagate_inbounds Base.getindex(t::AbstractStackedQTree, l, r, c) = t[l][r, c]
Base.@propagate_inbounds Base.getindex(t::AbstractStackedQTree, inds) = t[inds...]
Base.@propagate_inbounds Base.setindex!(t::AbstractStackedQTree, v, l, r, c) =  t[l][r, c] = v
Base.@propagate_inbounds Base.setindex!(t::AbstractStackedQTree, v, inds) = setindex!(t, v, inds...)
function Base.length(t::AbstractStackedQTree) end
Base.lastindex(t::AbstractStackedQTree) = length(t)
Base.size(t::AbstractStackedQTree) = length(t) > 0 ? size(t[1]) : (0,)
Base.broadcastable(t::AbstractStackedQTree) = Ref(t)

################ StackedQTree
struct StackedQTree{T <: AbstractVector{<:AbstractMatrix{UInt8}}} <: AbstractStackedQTree
    layers::T
end

StackedQTree(l::T) where T = StackedQTree{T}(l)
function StackedQTree(pic::AbstractMatrix{UInt8})
    m, n = size(pic)
    @assert m == n
    @assert isinteger(log2(m))
        
    l = Vector{typeof(pic)}()
    push!(l, pic)
    while size(l[end]) != (1, 1)
        m, n = size(l[end])
        push!(l, similar(pic, (m + 1) ÷ 2, (n + 1) ÷ 2))
    end
    StackedQTree(l)
end

function StackedQTree(pic::AbstractMatrix)
    pic = map(x -> x == 0 ? EMPTY : FULL, pic)
    StackedQTree(pic)
end

Base.@propagate_inbounds Base.getindex(t::StackedQTree, l::Integer) = t.layers[l]
Base.length(t::StackedQTree) = length(t.layers)

function buildqtree!(t::AbstractStackedQTree, layer=2)
    for l in layer:length(t)
        for r in 1:size(t[l], 1)
            for c in 1:size(t[l], 2)
                @inbounds qcode!(t, (l, r, c))
            end
        end
    end
end


################ ShiftedQTree
mutable struct PaddedMat{T <: AbstractMatrix{UInt8}} <: AbstractMatrix{UInt8}
    kernel::T
    size::Tuple{Int,Int}
    rshift::Int
    cshift::Int
    default::UInt8
end

function PaddedMat(l::T, sz::Tuple{Int,Int}=size(l), rshift=0, cshift=0; default=0x00) where {T <: AbstractMatrix{UInt8}}
    m = PaddedMat{T}(size(l), sz, rshift, cshift; default=default)
    m.kernel[2:end - 2, 2:end - 2] .= l
    m
end
function PaddedMat{T}(kernelsz::Tuple{Int,Int}, sz::Tuple{Int,Int}, 
    rshift=0, cshift=0; default=zero(T)) where {T <: AbstractMatrix{UInt8}}
    k = similar(T, kernelsz .+ 3) # +3 to keep top-down getindex in _collision_randbfs within the kernel bounds
    k[[1, end - 1, end], :] .= default
    k[:, [1, end - 1, end]] .= default
    PaddedMat(k, sz, rshift, cshift, default)
end

rshift!(l::PaddedMat, v) = l.rshift += v
cshift!(l::PaddedMat, v) = l.cshift += v
rshift(l::PaddedMat) = l.rshift
cshift(l::PaddedMat) = l.cshift
setrshift!(l::PaddedMat, v) = l.rshift = v
setcshift!(l::PaddedMat, v) = l.cshift = v
getrshift(l::PaddedMat) = l.rshift
getcshift(l::PaddedMat) = l.cshift
getshift(l::PaddedMat) = l.rshift, l.cshift
getdefault(l::PaddedMat) = l.default
function inkernelbounds(l::PaddedMat, r::Integer, c::Integer)
    if r <= l.rshift || c <= l.cshift
        return false
    end
    m, n = kernelsize(l)
    if r > m+l.rshift || c > n+l.cshift
        return false
    end
    true
end

inbounds(l::PaddedMat, r, c) = 0 < r <= size(l, 1) && 0 < c  <= size(l, 2)
kernelsize(l::PaddedMat) = size(l.kernel) .- 3
kernelsize(l::PaddedMat, i) = size(l.kernel, i) - 3
kernel(l::PaddedMat) = l.kernel
function Base.checkbounds(l::PaddedMat, I...) end # 关闭边界检查，允许负索引、超界索引
Base.@propagate_inbounds function Base.getindex(l::PaddedMat, r::Integer, c::Integer)
    if inkernelbounds(l, r, c) 
        return @inbounds l.kernel[r - l.rshift + 1, c - l.cshift + 1]
    else
        return l.default
    end
end
Base.@propagate_inbounds _getindex(l::PaddedMat, r::Integer, c::Integer) = l.kernel[r - l.rshift + 1, c - l.cshift + 1] #assume inkernelbounds
Base.@propagate_inbounds function Base.setindex!(l::PaddedMat, v, r::Integer, c::Integer)
    l.kernel[r - l.rshift + 1, c - l.cshift + 1] = v # kernel自身有边界检查
end

Base.size(l::PaddedMat) = l.size

struct ShiftedQTree{T <: AbstractVector{<:PaddedMat}} <: AbstractStackedQTree
    layers::T
    sizelevel::Int
end
const U8SQTree = ShiftedQTree{Vector{PaddedMat{Matrix{UInt8}}}}
ShiftedQTree(l::T) where T = ShiftedQTree{T}(l)
function ShiftedQTree(pic::PaddedMat{T}) where T
    sz = size(pic, 1)
    @assert size(pic, 1) == size(pic, 2)
    @assert isinteger(log2(sz))
    l = [pic]
    m, n = kernelsize(l[end])
    sizelevel = -1
    while sz != 1
        sz ÷= 2
        m, n = m ÷ 2 + 1, n ÷ 2 + 1
        push!(l, PaddedMat{T}((m, n), (sz, sz), default=getdefault(pic)))
        if sizelevel == -1 && m <= 2 && n <= 2
            sizelevel = length(l)
        end
    end
    if sizelevel != -1
        sizelevel = length(l)
    end
    ShiftedQTree(l, sizelevel)
end
function ShiftedQTree(pic::AbstractMatrix{UInt8}, sz::Integer; default=EMPTY)
    @assert isinteger(log2(sz))
    (sz >= size(pic, 1) && sz >= size(pic, 2)) || (@warn "ShiftedQTree cut off: $(size(pic))>$sz")
    ShiftedQTree(PaddedMat(pic, (sz, sz), default=default))
end
function ShiftedQTree(pic::AbstractMatrix{UInt8}; default=EMPTY)
    m = max(2, size(pic)...) # >2 for _collision_randbfs
    ShiftedQTree(pic, 2^ceil(Int, log2(m)), default=default)
end
function ShiftedQTree(pic::AbstractMatrix, args...; kargs...)
    @assert !isempty(pic)
    pic = map(x -> x == 0 ? EMPTY : FULL, pic)
    ShiftedQTree(pic, args...; kargs...)::U8SQTree
end
Base.@propagate_inbounds Base.getindex(t::ShiftedQTree, l::Integer) = t.layers[l]
Base.@propagate_inbounds function Base.getindex(t::ShiftedQTree, l, r, c)
    #if this function is called with @inbounds, that assume not only `inbounds` but also `inkernelbounds`
    @boundscheck return t[l][r, c]
    return @inbounds _getindex(t[l], r, c)
end

Base.length(t::ShiftedQTree) = length(t.layers)
sizelevel(t::ShiftedQTree) = t.sizelevel
inkernelbounds(t::ShiftedQTree, l, a, b) = inkernelbounds(t[l], a, b)
inkernelbounds(t::ShiftedQTree, ind) = inkernelbounds(t, ind...)
function buildqtree!(t::ShiftedQTree, layer=2)
    for l in layer:length(t)
        m = rshift(t[l - 1])
        n = cshift(t[l - 1])
        m2 = m ÷ 2
        n2 = n ÷ 2
        setrshift!(t[l], m2)
        setcshift!(t[l], n2)
        for r in 1:kernelsize(t[l])[1]
            for c in 1:kernelsize(t[l])[2]
#                 @show (l,m2+r,n2+c)
                @inbounds qcode!(t, (l, m2 + r, n2 + c))
            end
        end
    end
    t
end
function rshift!(t::ShiftedQTree, l::Integer, st::Integer)
    for i in l:-1:1
        rshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end
function cshift!(t::ShiftedQTree, l::Integer, st::Integer)
    for i in l:-1:1
        cshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end
function setrshift!(t::ShiftedQTree, l::Integer, st::Integer)
    for i in l:-1:1
        setrshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end
function setcshift!(t::ShiftedQTree, l::Integer, st::Integer)
    for i in l:-1:1
        setcshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end

function shift!(t::ShiftedQTree, l::Integer, st1::Integer, st2::Integer)
    for i in l:-1:1
        rshift!(t[i], st1)
        cshift!(t[i], st2)
        st1 *= 2
        st2 *= 2
    end
    buildqtree!(t, l + 1)
end
shift!(t::ShiftedQTree, l::Integer, st::Tuple{Integer,Integer}) = shift!(t, l, st...)
function setshift!(t::ShiftedQTree, l::Integer, st1::Integer, st2::Integer)
    for i in l:-1:1
        setrshift!(t[i], st1)
        setcshift!(t[i], st2)
        st1 *= 2
        st2 *= 2
    end
    buildqtree!(t, l + 1)
end
setshift!(t::ShiftedQTree, l::Integer, st::Tuple{Integer,Integer}) = setshift!(t, l, st...)
setshift!(t::ShiftedQTree, st::Tuple{Integer,Integer}) = setshift!(t, 1, st)
getshift(t::ShiftedQTree, l::Integer=1) = getshift(t[l])
kernelsize(t::ShiftedQTree, l::Integer=1) = kernelsize(t[l])
getcenter(t::ShiftedQTree) = getshift(t) .+ kernelsize(t) .÷ 2
getcenter(l::Integer, a::Integer, b::Integer) = indexcenter(l, a, b)
getcenter(ind::Tuple{Integer,Integer,Integer}) = getcenter(ind...)
center2lefttop(t::ShiftedQTree, center) = center .- kernelsize(t) .÷  2
setcenter!(t::ShiftedQTree, center) = setshift!(t, center2lefttop(t, center))
function inbounds(bgqt::ShiftedQTree, qt::ShiftedQTree)
    inbounds(bgqt[1], getcenter(qt)...)
end
function inkernelbounds(bgqt::ShiftedQTree, qt::ShiftedQTree)
    inkernelbounds(bgqt[1], getcenter(qt)...)
end
function outofbounds(bgqt::ShiftedQTree, qts)
    [i for (i, t) in enumerate(qts) if !inbounds(bgqt, t)]
end
function outofkernelbounds(bgqt::ShiftedQTree, qts)
    [i for (i, t) in enumerate(qts) if !inkernelbounds(bgqt, t)]
end

################ show
function charmat(mat; maxlen=49)
    m, n = size(mat)
    half = maxlen ÷ 2
    maxlen = half * 2 + 1
    if m <= maxlen && n <= maxlen
        return map(x -> 0 < x < 4 ? ("░", "▓", "▒")[x] : "▞", mat)
    elseif m > maxlen
        _n = min(maxlen, n) - 5
        if _n > 0
            _n2 = _n-_n÷2
            vsp = vcat(["⋮"], repeat([" "], _n2÷2), ["⋮"], repeat([" "], _n2-_n2÷2), ["⋱"], 
            repeat([" "], _n÷4), ["⋮"], repeat([" "], _n÷2-_n÷4), ["⋮"])
        else
            vsp = repeat(["⋮"], min(maxlen, n))
        end
        return vcat(charmat(@view(mat[1:half, :]), maxlen=maxlen),
            reshape(vsp, 1, min(maxlen, n)),
            charmat(@view(mat[end - half + 1:end, :]), maxlen=maxlen))
    else
        return hcat(charmat(@view(mat[:, 1:half]), maxlen=maxlen),
            reshape(repeat(["…"], m), m, 1),
            charmat(@view(mat[:, end - half + 1:end]), maxlen=maxlen))
    end
end
        
charimage(mat; kargs...) = join([join(l) * "\n" for l in eachrow(charmat(mat; kargs...))])
charimage(qt::ShiftedQTree; maxlen=49) = charimage(qt[max(1, ceil(Int, log2(size(qt[1], 1) / maxlen)) + 1)], maxlen=maxlen)
Base.show(io::IO, m::MIME"text/plain", mat::PaddedMat) = print(io, "PaddedMat $(size(mat)):\n", charimage(mat))
function Base.show(io::IO, m::MIME"text/plain", qt::ShiftedQTree)
    showlevel = max(1, ceil(Int, log2(size(qt[1], 1) / 49)) + 1)
    print(io, "ShiftedQTree{", length(qt))
    if length(qt) > 0
        print(io, " - ", size(qt[1]), "}  @level ", showlevel, " - ", size(qt[showlevel]), ":\n", charimage(qt[showlevel], maxlen=49))
    else
        print(io, "} {}")
    end
end

################ LinkedQTree
mutable struct QTreeNode{T}
    value::T
    parent::QTreeNode{T}
    children::SVector4{QTreeNode{T}}
    function QTreeNode{T}() where T
        n = new{T}()
        n.parent = n
        n.children = SVector4(n, n, n, n)
        n
    end
    QTreeNode{T}(v, p, c) where T = new{T}(v, p, c)
end
function QTreeNode{T}(value::T) where T
    n = QTreeNode{T}()
    n.value = value
    n
end
function QTreeNode{T}(parent::QTreeNode{T}) where T
    n = QTreeNode{T}()
    n.parent = parent
    n
end
function QTreeNode{T}(parent::QTreeNode{T}, value::T) where T
    n = QTreeNode{T}()
    n.parent = parent
    n.value = value
    n
end
QTreeNode(value::T) where T = QTreeNode{T}(value)
QTreeNode(value::T, parent, children) where T = QTreeNode{T}(value, parent, children)
QTreeNode(parent::QTreeNode{T}, args...) where T = QTreeNode{T}(parent, args...)
parent(n::QTreeNode) = n.parent
setparent!(n::QTreeNode, p::QTreeNode) = n.parent = p
children(n::QTreeNode) = n.children
Base.@propagate_inbounds child(n::QTreeNode, i) = n.children[i]
Base.@propagate_inbounds setchild!(n::QTreeNode, c::QTreeNode, i) = n.children[i] = c
isroot(n::QTreeNode) = n === parent(n)
isemptychild(n::QTreeNode, c::QTreeNode) = n === c
function Base.iterate(n::QTreeNode, Q=[n])
    isempty(Q) && return nothing
    n = popfirst!(Q)
    for c in children(n)
        (!isemptychild(n, c)) && push!(Q, c)
    end
    return n, Q
end

Base.eltype(::Type{QTreeNode{T}}) where T = QTreeNode{T}
CommonDatatypes.next(n::QTreeNode) = parent(n) # use as a linked list node
CommonDatatypes.setnext!(n::QTreeNode, next::QTreeNode) = setparent!(n, next)

################ HashSpacialQTree
abstract type AbstractSpacialQTree end
function Base.push!(t::AbstractSpacialQTree, ind::Index, label) end
function clear!(t::AbstractSpacialQTree, label::Int) end
function tree(t::AbstractSpacialQTree) end
Base.iterate(t::AbstractSpacialQTree, args...) = iterate(tree(t), args...)
function update!(t::AbstractSpacialQTree, inds, label::Int)
    clear!(t, label)
    for ind in inds
        push!(t, ind, label)
    end
end
struct HashSpacialQTree <: AbstractSpacialQTree
    qtree::Dict{Index, Vector{Int}}
end
HashSpacialQTree() = HashSpacialQTree(Dict{Index, Vector{Int}}())
Base.push!(t::HashSpacialQTree, ind::Index, label::Int) = push!(get!(Vector{Int}, t.qtree, ind), label)
Base.empty!(t::HashSpacialQTree) = (empty!(t.qtree); t)
tree(t::HashSpacialQTree) = t.qtree
Base.get(t::HashSpacialQTree, args...) = get(tree(t), args...)
Base.@propagate_inbounds Base.getindex(t::HashSpacialQTree, args...) = getindex(tree(t), args...)
Base.keys(t::HashSpacialQTree) = keys(tree(t))
Base.values(t::HashSpacialQTree) = values(tree(t))
Base.haskey(t::HashSpacialQTree, args...) = haskey(tree(t), args...)

################ LinkedSpacialQTree
mutable struct TreeNodeData
    index::Index
    list::DoublyLinkedList{ListNode{Int}}
    function TreeNodeData(index::Index)
        l = new()
        l.index = index
        l
    end
    TreeNodeData() = new()
end
const SpacialQTreeNode = QTreeNode{TreeNodeData}
new_spacial_qtree_node(index::Index) = QTreeNode(TreeNodeData(index))
new_spacial_qtree_node() = QTreeNode(TreeNodeData())
spacialindex(n::SpacialQTreeNode) = n.value.index
getnodelist(n::SpacialQTreeNode) = n.value.list
isemptynodelist(n::SpacialQTreeNode) = (!isdefined(n.value, :list)) || isempty(getnodelist(n))
function getnodelist!(node::SpacialQTreeNode)
    if !isdefined(node.value, :list)
        dl = DoublyLinkedList{ListNode{Int}}(Int(pointer_from_objref(node)))
        # @assert node == seektreenode(dl.tail)
        node.value.list = dl
    end
    node.value.list
end
struct LinkedSpacialQTree <: AbstractSpacialQTree
    qtree::SpacialQTreeNode
    label_to_nodes::IntMap{Vector{SVector4{ListNode{Int}}}}
    listnode_pool::LinkedList{ListNode{Int}}
    treenode_pool::LinkedList{SpacialQTreeNode}
end
LinkedSpacialQTree(index::Index, map) = LinkedSpacialQTree(new_spacial_qtree_node(index), map, LinkedList{ListNode{Int}}(), LinkedList{SpacialQTreeNode}())
LinkedSpacialQTree(map) = LinkedSpacialQTree(new_spacial_qtree_node(), map, LinkedList{ListNode{Int}}(), LinkedList{SpacialQTreeNode}())
tree(t::LinkedSpacialQTree) = t.qtree

isemptylabelnodes(t::LinkedSpacialQTree, label) = (!haskey(t.label_to_nodes, label)) || isempty(getlabelnodes(t, label))
getlabelnodes(t::LinkedSpacialQTree, label) = t.label_to_nodes[label]
function getlabelnodes!(t::LinkedSpacialQTree, label)
    if haskey(t.label_to_nodes, label)
        nodes = t.label_to_nodes[label]
    else
        nodes = SVector4{ListNode{Int}}()
        t.label_to_nodes[label] = nodes
    end
    nodes
end
function new_listnode_for_push(t::LinkedSpacialQTree, value::Int)
    cache = t.listnode_pool
    if isempty(cache)
        return newnode(cache, value)
    else
        n = popfirst!(cache)
        setvalue!(n, value)
        return n
    end
end
function new_spacial_qtree_node(t::LinkedSpacialQTree, parent::SpacialQTreeNode, index::Index)
    cache = t.treenode_pool
    if isempty(cache)
        # @show index
        return newnode(cache, parent, TreeNodeData(index))
    else
        # @show length(cache)
        n = popfirst!(cache)
        # @assert n.children == [n,n,n,n]
        setparent!(n, parent)
        n.value.index = index
        return n
    end
end
function remove_tree_node(t::LinkedSpacialQTree, node::SpacialQTreeNode)
    p = parent(node)
    ind = spacialindex(node)
    # @show ind
    setchild!(p, p, childnumber(ind)+1)
    pushfirst!(t.treenode_pool, node)
end
function Base.push!(t::LinkedSpacialQTree, ind::Index, label::Int)
    # @show ind, label
    tn = t.qtree
    aind = spacialindex(tn)
    # @show spacialindex(tn)
    if inrange(aind, ind)
        while true
            aind = spacialindex(tn)
            aind[1] <= ind[1] && break
            cn = childnumber(aind, ind)
            # @show aind, ind, cn
            cnode = child(tn, cn+1)
            if isemptychild(tn, cnode)
                cnode = new_spacial_qtree_node(t, tn, child(aind, cn))
                setchild!(tn, cnode, cn+1)
            end
            tn = cnode
        end
        # @assert aind == ind
        n = new_listnode_for_push(t, label)
        pushfirst!(getnodelist!(tn), n)
        push!(getlabelnodes!(t, label), n)
    end
end

function seektreenode(listnode::ListNode{Int})
    head = seekhead(listnode)
    unsafe_pointer_to_objref(Ptr{Any}(value(head)))::SpacialQTreeNode
end
function clear!(t::LinkedSpacialQTree, label::Int)
    if !isemptylabelnodes(t, label)
        nodes = getlabelnodes(t, label)
        for n in nodes
            pushfirst!(t.listnode_pool, pop!(n))
        end
        empty!(nodes)
    end
end

collect_labels(t::QTreeNode) = collect(getnodelist(t))
collect_labels(filter, t::QTreeNode) = collect(filter, getnodelist(t))
function collect_tree(t::LinkedSpacialQTree)
    hashtree = Dict{Index, Vector{Int}}()
    for n in tree(t)
        spinds = spacialindex(n)
        if !isemptynodelist(n)
            lbs = collect_labels(n)
            hashtree[spinds] = lbs
        end
    end
    return hashtree
end
include("qtree_functions.jl")
end
