module LinkedList
export DoubleList, ListNode, movetofirst!, IntMap, ishead, istail, seek_tail
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
function Base.iterate(l::ListNode, p=l)
    istail(p) ?  nothing : (p.value, p.next)
end
Base.IteratorSize(::Type{<:ListNode}) = Base.SizeUnknown()
Base.eltype(::Type{ListNode{T}}) where T = T
function seek_tail(l::ListNode)
    while !istail(l)
        l = l.next
    end
    l
end

mutable struct DoubleList{T}
    head::ListNode{T}
    tail::ListNode{T}
end
function DoubleList{T}() where T
    h = ListNode{T}()
    t = ListNode{T}()
    h.next = t
    t.prev = h
    DoubleList(h, t)
end
function DoubleList{T}(hv::T, tv::T) where T
    l = DoubleList{T}()
    l.head.value = hv
    l.tail.value = tv
    l
end
ishead(n::ListNode) = n.prev === n
istail(n::ListNode) = n.next === n
Base.isempty(l::DoubleList) = l.head.next === l.tail
function Base.pushfirst!(l::DoubleList, n::ListNode)
    n.next = l.head.next
    n.prev = l.head
    l.head.next = n
    n.next.prev = n
    n
end
function Base.pop!(n::ListNode)
    n.prev.next = n.next
    n.next.prev = n.prev
    n
end
Base.pop!(l::DoubleList, n::ListNode) = Base.pop!(n)
Base.popfirst!(l::DoubleList) = (@assert !isempty(l); pop!(l.head.next))

function movetofirst!(l::DoubleList, n::ListNode)
    pop!(l, n)
    pushfirst!(l, n)
end
function Base.iterate(l::DoubleList, p=l.head.next)
    p === l.tail && return nothing
    p.value, p.next
end
Base.IteratorSize(::Type{<:DoubleList}) = Base.SizeUnknown()
Base.eltype(::Type{DoubleList{T}}) where T = T

function collect!(l::DoubleList, collection)
    p = l.head.next
    while p !== l.tail
        push!(collection, p.value)
        p = p.next
    end
    collection
end
function collect!(l::DoubleList, collection, firstn)
    p = l.head.next
    for i in 1:firstn
        if p === l.tail
            break
        end
        push!(collection, p.value)
        p = p.next
    end
    collection
end
function collect!(filter, l::DoubleList, collection)
    p = l.head.next
    while p !== l.tail
        v = p.value
        filter(v) && push!(collection, v)
        p = p.next
    end
    collection
end
function collect!(filter, l::DoubleList, collection, firstn)
    p = l.head.next
    for i in 1:firstn
        if p === l.tail
            break
        end
        v = p.value
        filter(v) && push!(collection, v)
        p = p.next
    end
    collection
end
Base.collect(l::DoubleList{T}, args...) where T = collect!(l, Vector{T}(), args...)
Base.collect(filter, l::DoubleList{T}, args...) where T = collect!(filter, l, Vector{T}(), args...)
struct IntMap{T}
    map::T
end
Base.haskey(im::IntMap, key) = isassigned(im.map, key)
Base.getindex(im::IntMap, ind...) = getindex(im.map, ind...)
Base.setindex!(im::IntMap, v, ind...) = setindex!(im.map, v, ind...)

end
