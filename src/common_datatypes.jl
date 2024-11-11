module CommonDatatypes
export ListNode, LinkedList, DoublyLinkedList, IntMap, SVector4, IntSet,
    newnode, next, prev, value, setnext!, setprev!, setvalue!, ishead, istail, seekhead, seektail, movetofirst!

##### ListNode
mutable struct ListNode{T}
    value::T
    prev::ListNode{T}
    next::ListNode{T}
    function ListNode{T}() where T
        n = new{T}()
        n.prev = n
        n.next = n
        n
    end
end
function ListNode{T}(value::T) where T
    n = ListNode{T}()
    n.value = value
    n
end
ListNode(value::T) where T = ListNode{T}(value)
next(n::ListNode) = n.next
prev(n::ListNode) = n.prev
value(n::ListNode) = n.value
setnext!(n::ListNode, nn) = n.next = nn
setprev!(n::ListNode, nn) = n.prev = nn
setvalue!(n::ListNode, v) = n.value = v
ishead(n::ListNode) = prev(n) === n
istail(n::ListNode) = next(n) === n
function Base.iterate(l::ListNode, p=l) # from this node to the end
    istail(p) ? nothing : (value(p), next(p))
end
Base.IteratorSize(::Type{<:ListNode}) = Base.SizeUnknown()
Base.eltype(::Type{ListNode{T}}) where T = T
function seektail(n::ListNode)
    while !istail(n)
        n = next(n)
    end
    n
end
function seekhead(n::ListNode)
    while !ishead(n)
        n = prev(n)
    end
    n
end
function Base.pop!(n::ListNode)
    n.prev.next = next(n)
    n.next.prev = prev(n)
    n
end

##### LinkedList
mutable struct LinkedList{NodeType}
    head::NodeType
end
function LinkedList{T}() where T
    h = T()
    LinkedList(h)
end
newnode(l::LinkedList{T}, args...) where T = T(args...)
head(l::LinkedList) = l.head
ishead(l::LinkedList, n) = head(l) === n
Base.isempty(l::LinkedList) = ishead(l, next(head(l))) # doesn't have a tail sentinel
function Base.pushfirst!(l::LinkedList{T}, n::T) where T
    h = head(l)
    setnext!(n, next(h))
    setnext!(h, n)
    n
end
Base.pushfirst!(l::LinkedList, v) = pushfirst!(l, newnode(l, v))
function Base.popfirst!(l::LinkedList)
    @assert !isempty(l)
    h = head(l)
    n = next(h)
    setnext!(h, next(n))
    n
end
Base.IteratorSize(::Type{<:LinkedList}) = Base.SizeUnknown()
Base.eltype(::Type{LinkedList{T}}) where T = eltype(T)
function Base.iterate(l::LinkedList, p=next(head(l)))
    ishead(l, p) ? nothing : (value(p), next(p))
end

##### DoublyLinkedList
mutable struct DoublyLinkedList{NodeType}
    head::NodeType
    function DoublyLinkedList{T}() where T
        l = new()
        h = T()
        t = T()
        setnext!(h, t)
        setprev!(t, h)
        l.head = h
        l
    end
end
function DoublyLinkedList{T}(hv, tv) where T
    l = DoublyLinkedList{T}()
    setvalue!(l|>head, hv)
    setvalue!(l|>head|>next, tv)
    l
end
function DoublyLinkedList{T}(hv) where T
    l = DoublyLinkedList{T}()
    setvalue!(l|>head, hv)
    l
end
newnode(l::DoublyLinkedList{T}, args...) where T = T(args...)
head(l::DoublyLinkedList) = l.head
Base.isempty(l::DoublyLinkedList) = istail(next(head(l)))
function Base.pushfirst!(l::DoublyLinkedList{T}, n::T) where T
    h = head(l)
    hn = next(h)
    setnext!(n, hn)
    setprev!(n, h)
    setnext!(h, n)
    setprev!(hn, n)
    n
end
Base.pushfirst!(l::DoublyLinkedList, v) = pushfirst!(l, newnode(l, v))
Base.pop!(l::DoublyLinkedList, n) = pop!(n)
Base.popfirst!(l::DoublyLinkedList) = (@assert !isempty(l); pop!(next(head(l))))

function movetofirst!(l::DoublyLinkedList, n)
    pop!(l, n)
    pushfirst!(l, n)
end
Base.iterate(l::DoublyLinkedList, args...) = iterate(next(head(l)), args...)
Base.IteratorSize(::Type{<:DoublyLinkedList}) = Base.SizeUnknown()
Base.eltype(::Type{DoublyLinkedList{T}}) where T = eltype(T)

function collect!(l::DoublyLinkedList, collection)
    p = next(head(l))
    while !istail(p)
        push!(collection, value(p))
        p = next(p)
    end
    # @assert p === l.tail
    collection
end
function collect!(l::DoublyLinkedList, collection, firstn)
    p = next(head(l))
    for i in 1:firstn
        if istail(p)
            # @assert p === l.tail
            break
        end
        push!(collection, value(p))
        p = next(p)
    end
    collection
end
function collect!(filter, l::DoublyLinkedList, collection)
    p = next(head(l))
    while !istail(p)
        v = value(p)
        filter(v) && push!(collection, v)
        p = next(p)
    end
    # @assert p === l.tail
    collection
end
function collect!(filter, l::DoublyLinkedList, collection, firstn)
    p = next(head(l))
    for i in 1:firstn
        if istail(p)
            # @assert p === l.tail
            break
        end
        v = value(p)
        filter(v) && push!(collection, v)
        p = next(p)
    end
    collection
end
Base.collect(l::DoublyLinkedList, args...) = collect!(l, Vector{eltype(l)}(), args...)
Base.collect(filter, l::DoublyLinkedList, args...) = collect!(filter, l, Vector{eltype(l)}(), args...)

##### IntMap
struct IntMap{T}
    map::T
end
Base.haskey(im::IntMap, key) = isassigned(im.map, key)
Base.getindex(im::IntMap, ind...) = getindex(im.map, ind...)
Base.setindex!(im::IntMap, v, ind...) = setindex!(im.map, v, ind...)

##### SVector4
mutable struct SVector4Core{T} 
    e1::T
    e2::T
    e3::T
    e4::T
    SVector4Core{T}() where T = new()
end
mutable struct SVector4{T} 
    vec::SVector4Core{T}
    len::Int
end
SVector4{T}() where T = SVector4{T}(SVector4Core{T}(), 0)
function SVector4{T}(e1, e2, e3, e4) where T
    v = SVector4{T}()
    v.vec.e1 = e1
    v.vec.e2 = e2
    v.vec.e3 = e3
    v.vec.e4 = e4
    v.len = 4
    v
end
SVector4(e1::T, e2::T, e3::T, e4::T) where T = SVector4{T}(e1, e2, e3, e4)
Base.getindex(v::SVector4{T}, i) where T = getfield(v.vec, i)::T
Base.setindex!(v::SVector4{T}, x::T, i) where T = setfield!(v.vec, i, x)
Base.push!(v::SVector4{T}, x::T) where T = v[v.len += 1] = x
Base.length(v::SVector4) = v.len
Base.iterate(v::SVector4, i=1) = i <= length(v) ? (v[i], i+1) : nothing
Base.eltype(::Type{SVector4{T}}) where T = T
Base.empty!(v::SVector4) = (v.len = 0)
Base.isempty(v::SVector4) = v.len == 0

##### IntSet
mutable struct IntSet <: AbstractSet{Int}
    list::Vector{Int}
    slots::BitVector
end
IntSet(len::Integer, ::Val{:empty}) = IntSet(Vector{Int}(), falses(len))
IntSet(len::Integer, ::Val{:full}) = IntSet(Vector{Int}(1:len), trues(len))
IntSet(len::Integer) = IntSet(len, Val(:empty))
function Base.push!(s::IntSet, x::Int)
    s.slots[x] || (push!(s.list, x); s.slots[x] = true)
    s
end
function Base.empty!(s::IntSet)
    empty!(s.list)
    fill!(s.slots, false)
    s
end
# pop! is not allowed
Base.iterate(s::IntSet, args...) = iterate(s.list, args...)
Base.length(s::IntSet) = length(s.list)
Base.in(x::Int, s::IntSet) = s.slots[x]

end
