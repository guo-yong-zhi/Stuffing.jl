module LinkedList
export DoubleList, ListNode, movetofirst!, IntMap, next, prev, value, setvalue!, ishead, istail, seek_head, seek_tail
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

mutable struct DoubleList{T}
    head::ListNode{T}
    # tail::ListNode{T}
end
function DoubleList{T}() where T
    h = ListNode{T}()
    t = ListNode{T}()
    h.next = t
    t.prev = h
    DoubleList(h)
end
function DoubleList{T}(hv::T, tv::T) where T
    l = DoubleList{T}()
    l.head.value = hv
    l.head.next.value = tv
    l
end
function DoubleList{T}(hv::T) where T
    l = DoubleList{T}()
    l.head.value = hv
    l
end
head(l::DoubleList) = l.head
Base.isempty(l::DoubleList) = istail(next(head(l)))
function Base.pushfirst!(l::DoubleList, n::ListNode)
    h = head(l)
    hn = next(h)
    n.next = hn
    n.prev = h
    h.next = n
    hn.prev = n
    n
end

Base.pop!(l::DoubleList, n::ListNode) = pop!(n)
Base.popfirst!(l::DoubleList) = (@assert !isempty(l); pop!(next(head(l))))

function movetofirst!(l::DoubleList, n::ListNode)
    pop!(l, n)
    pushfirst!(l, n)
end
Base.iterate(l::DoubleList, args...) = iterate(next(head(l)), args...)
Base.IteratorSize(::Type{<:DoubleList}) = Base.SizeUnknown()
Base.eltype(::Type{DoubleList{T}}) where T = T

function collect!(l::DoubleList, collection)
    p = next(head(l))
    while !istail(p)
        push!(collection, value(p))
        p = next(p)
    end
    # @assert p === l.tail
    collection
end
function collect!(l::DoubleList, collection, firstn)
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
function collect!(filter, l::DoubleList, collection)
    p = next(head(l))
    while !istail(p)
        v = value(p)
        filter(v) && push!(collection, v)
        p = next(p)
    end
    # @assert p === l.tail
    collection
end
function collect!(filter, l::DoubleList, collection, firstn)
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
Base.collect(l::DoubleList{T}, args...) where T = collect!(l, Vector{T}(), args...)
Base.collect(filter, l::DoubleList{T}, args...) where T = collect!(filter, l, Vector{T}(), args...)
struct IntMap{T}
    map::T
end
Base.haskey(im::IntMap, key) = isassigned(im.map, key)
Base.getindex(im::IntMap, ind...) = getindex(im.map, ind...)
Base.setindex!(im::IntMap, v, ind...) = setindex!(im.map, v, ind...)

end
