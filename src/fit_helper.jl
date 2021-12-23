using ..LinkedList
abstract type AbstractOptimiser end
function apply(o::AbstractOptimiser, x, Δ) end
function apply!(o::AbstractOptimiser, x, Δ)
    Δ .= apply(o, x, Δ)
end
(opt::AbstractOptimiser)(x, Δ) = apply(opt::AbstractOptimiser, x, Δ)
Base.broadcastable(o::AbstractOptimiser) = Ref(o)
reset!(o, x) = nothing
eta(o) = NaN
eta!(o, e) = NaN

mutable struct Momentum <: AbstractOptimiser
    eta::Float64
    rho::Float64
    velocity::IdDict
end
Momentum(η, ρ=0.5) = Momentum(η, ρ, IdDict())
Momentum(;η=0.25, ρ=0.5) = Momentum(η, ρ, IdDict())
function apply(o::Momentum, x, Δ)
    η, ρ = o.eta, o.rho
    Δ = collect(Float64, Δ)
    v = get!(o.velocity, x, Δ)
    @. v = ρ * v + (1 - ρ) * Δ
    η .* v
end
reset!(o::Momentum, x) =  pop!(o.velocity, x)
eta(o::Momentum) = o.eta
eta!(o::Momentum, e) = o.eta = e

mutable struct SGD <: AbstractOptimiser
    eta::Float64
end
SGD(;η=0.25) = SGD(η)
apply(o::SGD, x, Δ) = o.eta .* Δ
reset!(o::SGD, x) =  nothing
eta(o::SGD) = o.eta
eta!(o::SGD, e) = o.eta = e


@assert QTrees.EMPTY == 1 && QTrees.FULL == 2 && QTrees.MIX == 3
@inline decode2(c) = @inbounds (0, 2, 1)[c]

# const DECODETABLE = [0, 2, 1]
# near(a::Integer, b::Integer, r=1) = a-r:a+r, b-r:b+r
# near(m::AbstractMatrix, a::Integer, b::Integer, r=1) = @view m[near(a, b, r)...]
# const KERNEL = collect.(Iterators.product(-1:1,-1:1))
# gard2d(m::AbstractMatrix) = sum(KERNEL .* m)
# gard2d(t::ShiftedQTree, l, a, b) = gard2d(decode2(near(t[l],a,b)))|>Tuple

function gard2d(t::ShiftedQTree, l, a, b) # FULL is white, Positive directions are right & down 
    m = t[l]
    diag = -decode2(m[a - 1, b - 1]) + decode2(m[a + 1, b + 1])
    cdiag = -decode2(m[a - 1, b + 1]) + decode2(m[a + 1, b - 1])
    (
    + diag
    + cdiag
    - decode2(m[a - 1, b])
    + decode2(m[a + 1, b])
    ), (
    + diag
    - cdiag
    - decode2(m[a, b - 1])
    + decode2(m[a, b + 1])
    ) # (h, w)
end

# function intlog2(x::Float64) #not safe, x can't be nan or inf
#     #Float64 符号位(S)，编号63；阶码位，编号62 ~52
#     b64 = reinterpret(UInt64, x)
#     m = UInt64(0x01)<<63 #符号位mask
#     Int(1-((b64&m)>>62)), Int((b64&(~m)) >> 52 - 1023) #符号位:1-2S (1->-1、0->1)，指数位 - 1023
# end
function intlog2(x::Float64) # not safe, x>0 and x can't be nan or inf
    # Float64 符号位(S)，编号63；阶码位，编号62 ~52
    b64 = reinterpret(Int64, x)
    (b64 >> 52 - 1023) # 符号位:1-2S (1->-1、0->1)，指数位 - 1023
end

struct LRU{T,MAPTYPE}
    list::DoubleList{T}
    map::MAPTYPE
end
LRU{T}() where T = LRU{T,Dict}(DoubleList{T}(), Dict())
LRU{T}(map::U) where {T,U} = LRU{T,U}(DoubleList{T}(), map)

function Base.push!(lru::LRU, v)
    if haskey(lru.map, v)
        n = lru.map[v]
        movetofirst!(lru.list, n)
    else
        n = ListNode(v)
        lru.map[v] = n
        pushfirst!(lru.list, n)
    end
    v
end
take!(lru::LRU, args...) = LinkedList.take!(lru.list, args...)
take(lru::LRU, args...) = LinkedList.take(lru.list, args...)
Base.broadcastable(lru::LRU) = Ref(lru)

intlru(n) = LRU{Int}(IntMap(Vector{ListNode{Int}}(undef, n)))

mutable struct MonotoneIndicator{T}
    min::T
    age::Int
end
MonotoneIndicator{T}() where T = MonotoneIndicator{T}(typemax(T), 0)
function reset!(i::MonotoneIndicator{T}) where T
    i.min = typemax(T)
    i.age = 0
end
resetage!(i::MonotoneIndicator) = i.age = 0
function update!(i::MonotoneIndicator{T}, v) where T
    i.age += 1
    if v < i.min
        i.age = 0
        i.min = v
    end
end