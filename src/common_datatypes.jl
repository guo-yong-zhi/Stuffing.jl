module CommonDatatypes
export DoublyLinkedList, ListNode, IntMap, SVector4, IntSet, movetofirst!, next, prev, value, setvalue!, ishead, istail, seek_head, seek_tail
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
function seek_tail(n::ListNode)
    while !istail(n)
        n = next(n)
    end
    n
end
function seek_head(n::ListNode)
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

mutable struct DoublyLinkedList{T}
    head::T
end
function DoublyLinkedList{T}() where T
    h = T()
    t = T()
    setnext!(h, t)
    setprev!(t, h)
    DoublyLinkedList(h)
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
head(l::DoublyLinkedList) = l.head
Base.isempty(l::DoublyLinkedList) = istail(next(head(l)))
function Base.pushfirst!(l::DoublyLinkedList, n)
    h = head(l)
    hn = next(h)
    setnext!(n, hn)
    setprev!(n, h)
    setnext!(h, n)
    setprev!(hn, n)
    n
end

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
Base.collect(l::DoublyLinkedList, args...) where T = collect!(l, Vector{eltype(l)}(), args...)
Base.collect(filter, l::DoublyLinkedList, args...) where T = collect!(filter, l, Vector{eltype(l)}(), args...)

struct IntMap{T}
    map::T
end
Base.haskey(im::IntMap, key) = isassigned(im.map, key)
Base.getindex(im::IntMap, ind...) = getindex(im.map, ind...)
Base.setindex!(im::IntMap, v, ind...) = setindex!(im.map, v, ind...)

mutable struct SVector4{T} 
    e1::T
    e2::T
    e3::T
    e4::T
    len::Int
    SVector4{T}() where T = (v = new(); v.len = 0; v)
end
function SVector4{T}(e1, e2, e3, e4) where T
    v = SVector4{T}()
    v.e1 = e1
    v.e2 = e2
    v.e3 = e3
    v.e4 = e4
    v.len = 4
    v
end
Base.getindex(v::SVector4, i) = getfield(v, i)
Base.setindex!(v::SVector4, x, i) = setfield!(v, i, x)
Base.push!(v::SVector4, x) = v[v.len += 1] = x
Base.length(v::SVector4) = v.len
Base.iterate(v::SVector4, i=1) = i <= length(v) ? (v[i], i+1) : nothing
Base.eltype(::Type{SVector4{T}}) where T = T

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
