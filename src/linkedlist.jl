module LinkedList
export DoubleList, ListNode, movetofirst!, IntMap
mutable struct ListNode{T}
    value::T
    prev::ListNode
    next::ListNode
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
function movetofirst!(l::DoubleList, n::ListNode)
    pop!(l, n)
    pushfirst!(l, n)
end

function take!(l::DoubleList, collection)
    p = l.head.next
    while p !== l.tail
        push!(collection, p.value)
        p = p.next
    end
    collection
end
function take!(l::DoubleList, collection, firstn)
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
function take(l::DoubleList{T}, args...) where T
    collection = Vector{T}()
    take!(l, collection, args...)
end

struct IntMap{T}
    map::T
end
Base.haskey(im::IntMap, key) = isassigned(im.map, key)
Base.getindex(im::IntMap, ind...) = getindex(im.map, ind...)
Base.setindex!(im::IntMap, v, ind...) = setindex!(im.map, v, ind...)

end