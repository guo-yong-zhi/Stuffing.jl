module QTree
export AbstractStackQtree, StackQtree, ShiftedQtree, buildqtree!,
    shift!, setrshift!,　setcshift!, setshift!, getshift, getcenter, setcenter!,
    collision, collision_bfs_rand, batchcollision,
    findroom_uniform, findroom_gathering, levelnum, outofbounds, kernelsize, 
    placement!, overlap, overlap!, decode, charimage

using Random

const PERM4 = ((1, 2, 3, 4), (1, 2, 4, 3), (1, 3, 2, 4), (1, 3, 4, 2), (1, 4, 2, 3), (1, 4, 3, 2), (2, 1, 3, 4), 
(2, 1, 4, 3), (2, 3, 1, 4), (2, 3, 4, 1), (2, 4, 1, 3), (2, 4, 3, 1), (3, 1, 2, 4), (3, 1, 4, 2), (3, 2, 1, 4), 
(3, 2, 4, 1), (3, 4, 1, 2), (3, 4, 2, 1), (4, 1, 2, 3), (4, 1, 3, 2), (4, 2, 1, 3), (4, 2, 3, 1), (4, 3, 1, 2), (4, 3, 2, 1))
@assert length(PERM4) == 24
@inline shuffle4() = @inbounds PERM4[rand(1:24)]
@inline function child(ind::Tuple{Int,Int,Int}, n::Int)
    @inbounds (ind[1] - 1, 2ind[2] - n & 0x01, 2ind[3] - (n & 0x02) >> 1)
end
@inline parent(ind::Tuple{Int,Int,Int}) = @inbounds (ind[1] + 1, (ind[2] + 1) ÷ 2, (ind[3] + 1) ÷ 2)
indexcenter(l::Integer, a::Integer, b::Integer) = l == 1 ? (a, b) : (2^(l-1)*(a-1)+2^(l-2), 2^(l-1)*(b-1)+2^(l-2))
indexcenter(ind) = indexcenter(ind...)
function indexrange(l::Integer, a::Integer, b::Integer)
    r = 2 ^ (l - 1)
    r*(a-1)+1:r*a, r*(b-1)+1:r*b
end
indexrange(ind) = indexrange(ind...)
@inline function qcode(Q, i)
    _getindex(Q, child(i, 1)) | _getindex(Q, child(i, 2)) | _getindex(Q, child(i, 3)) | _getindex(Q, child(i, 4))
end
@inline qcode!(Q, i) = _setindex!(Q, qcode(Q, i), i)
decode(c) = (0., 1., 0.5)[c]

const FULL = 0x02; EMPTY = 0x01; MIX = 0x03

abstract type AbstractStackQtree end
function Base.getindex(t::AbstractStackQtree, l::Integer) end
Base.getindex(t::AbstractStackQtree, l, r, c) = t[l][r, c]
Base.getindex(t::AbstractStackQtree, inds::Tuple{Int,Int,Int}) = t[inds...]
Base.setindex!(t::AbstractStackQtree, v, l, r, c) =  t[l][r, c] = v
Base.setindex!(t::AbstractStackQtree, v, inds::Tuple{Int,Int,Int}) = t[inds...] = v
@inline _getindex(t::AbstractStackQtree, inds) = _getindex(t, inds...)
@inline _getindex(t::AbstractStackQtree, l, r, c) = @inbounds _getindex(t[l], r, c)
@inline _setindex!(t::AbstractStackQtree, v, inds) = _setindex!(t, v, inds...)
@inline _setindex!(t::AbstractStackQtree, v, l, r, c) = @inbounds t[l][r, c] = v
#Base.setindex!中调用@boundscheck时用@inline无法成功内联致@propagate_inbounds失效，
#无法传递上层的@inbounds，故专门实现一个inbounds版的_setindex!
function levelnum(t::AbstractStackQtree) end
Base.lastindex(t::AbstractStackQtree) = levelnum(t)
Base.size(t::AbstractStackQtree) = levelnum(t) > 0 ? size(t[1]) : (0,)
Base.broadcastable(t::AbstractStackQtree) = Ref(t)

################ StackQtree
struct StackQtree{T <: AbstractVector{<:AbstractMatrix{UInt8}}} <: AbstractStackQtree
    layers::T
end

StackQtree(l::T) where T = StackQtree{T}(l)
function StackQtree(pic::AbstractMatrix{UInt8})
    m, n = size(pic)
    @assert m == n
    @assert isinteger(log2(m))
        
    l = Vector{typeof(pic)}()
    push!(l, pic)
    while size(l[end]) != (1, 1)
        m, n = size(l[end])
        push!(l, similar(pic, (m + 1) ÷ 2, (n + 1) ÷ 2))
    end
    StackQtree(l)
end

function StackQtree(pic::AbstractMatrix)
    pic = map(x -> x == 0 ? EMPTY : FULL, pic)
    StackQtree(pic)
end

Base.getindex(t::StackQtree, l::Integer) = t.layers[l]
levelnum(t::StackQtree) = length(t.layers)

function buildqtree!(t::AbstractStackQtree, layer=2)
    for l in layer:levelnum(t)
        for r in 1:size(t[l], 1)
            for c in 1:size(t[l], 2)
                qcode!(t, (l, r, c))
            end
        end
    end
end


################ ShiftedQtree
mutable struct PaddedMat{T <: AbstractMatrix{UInt8}} <: AbstractMatrix{UInt8}
    kernel::T
    size::Tuple{Int,Int}
    rshift::Int
    cshift::Int
    default::UInt8
end

function PaddedMat(l::T, sz::Tuple{Int,Int}=size(l), rshift=0, cshift=0; default=0x00) where {T <: AbstractMatrix{UInt8}}
    m = PaddedMat{T}(size(l), sz, rshift, cshift; default=default)
    m.kernel[2:end-2, 2:end-2] .= l
    m
end
function PaddedMat{T}(kernelsz::Tuple{Int,Int}, sz::Tuple{Int,Int}=size(l), 
    rshift=0, cshift=0; default=0x00) where {T <: AbstractMatrix{UInt8}}
    k = similar(T, kernelsz.+3)
    k[[1, end-1, end], :] .= default
    k[:, [1, end-1, end]] .= default
    PaddedMat(k, sz, rshift, cshift, default)
end
# PaddedMat{T}(l::T, sz::Tuple{Int,Int}=size(l).-2, rshift=0, 
# cshift=0; default=0x00) where {T <: AbstractMatrix{UInt8}} = PaddedMat(l, sz, rshift, cshift; default=default)

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
function inkernelbounds(l::PaddedMat, r, c)
    r -= l.rshift
    c -= l.cshift
    if r <= 0 || c <= 0
        return false
    end
    m, n = kernelsize(l)
    if r > m || c > n
        return false
    end
    true
end
inbounds(l::PaddedMat, r, c) = 0 < r <= size(l, 1) && 0 < c  <= size(l, 2)
kernelsize(l::PaddedMat) = size(l.kernel).-3
kernelsize(l::PaddedMat, i) = size(l.kernel, i)-3
kernel(l::PaddedMat) = l.kernel
function Base.checkbounds(l::PaddedMat, I...) end #关闭边界检查，允许负索引、超界索引
function Base.getindex(l::PaddedMat, r, c)
    if inkernelbounds(l, r, c)
        return @inbounds l.kernel[r - l.rshift + 1, c - l.cshift + 1]
    end
    return l.default
end
function _getindex(l::PaddedMat, r, c)
    return @inbounds l.kernel[r - l.rshift + 1, c - l.cshift + 1]
end
Base.@propagate_inbounds function Base.setindex!(l::PaddedMat, v, r, c)
    l.kernel[r - l.rshift + 1, c - l.cshift + 1] = v #kernel自身有边界检查
end

Base.size(l::PaddedMat) = l.size

struct ShiftedQtree{T <: AbstractVector{<:PaddedMat}} <: AbstractStackQtree
    layers::T
end

ShiftedQtree(l::T) where T = ShiftedQtree{T}(l)
function ShiftedQtree(pic::PaddedMat{T}) where T
    sz = size(pic, 1)
    @assert size(pic, 1) == size(pic, 2)
    @assert isinteger(log2(sz))
    l = [pic]
    m, n = kernelsize(l[end])
    while sz != 1
        sz ÷= 2
        m, n = m ÷ 2 + 1, n ÷ 2 + 1
#         @show m,n
        push!(l, PaddedMat{T}((m, n), (sz, sz), default=getdefault(pic)))
    end
    ShiftedQtree(l)
end
function ShiftedQtree(pic::AbstractMatrix{UInt8}, sz::Integer; default=EMPTY)
    @assert isinteger(log2(sz))
    ShiftedQtree(PaddedMat(pic, (sz, sz), default=default))
end
function ShiftedQtree(pic::AbstractMatrix{UInt8}; default=EMPTY)
    m = max(size(pic)...)
    ShiftedQtree(pic, 2^ceil(Int, log2(m)), default=default)
end
function ShiftedQtree(pic::AbstractMatrix, args...; kargs...)
    @assert !isempty(pic)
    pic = map(x -> x == 0 ? EMPTY : FULL, pic)
    ShiftedQtree(pic, args...; kargs...)
end
Base.@propagate_inbounds Base.getindex(t::ShiftedQtree, l::Integer) = t.layers[l]
levelnum(t::ShiftedQtree) = length(t.layers)
function buildqtree!(t::ShiftedQtree, layer=2)
    for l in layer:levelnum(t)
        m = rshift(t[l - 1])
        n = cshift(t[l - 1])
        m2 = floor(Int, m/2)
        n2 = floor(Int, n/2)
        setrshift!(t[l], m2)
        setcshift!(t[l], n2)
        for r in 1:kernelsize(t[l])[1]
            for c in 1:kernelsize(t[l])[2]
#                 @show (l,m2+r,n2+c)
                qcode!(t, (l, m2 + r, n2 + c))
            end
        end
    end
    t
end
function rshift!(t::ShiftedQtree, l::Integer, st::Integer)
    for i in l:-1:1
        rshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end
function cshift!(t::ShiftedQtree, l::Integer, st::Integer)
    for i in l:-1:1
        cshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end
function setrshift!(t::ShiftedQtree, l::Integer, st::Integer)
    for i in l:-1:1
        setrshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end
function setcshift!(t::ShiftedQtree, l::Integer, st::Integer)
    for i in l:-1:1
        setcshift!(t[i], st)
        st *= 2
    end
    buildqtree!(t, l + 1)
end

function shift!(t::ShiftedQtree, l::Integer, st1::Integer, st2::Integer)
    for i in l:-1:1
        rshift!(t[i], st1)
        cshift!(t[i], st2)
        st1 *= 2
        st2 *= 2
    end
    buildqtree!(t, l + 1)
end
shift!(t::ShiftedQtree, l::Integer, st::Tuple{Integer, Integer}) = shift!(t, l, st...)
function setshift!(t::ShiftedQtree, l::Integer, st1::Integer, st2::Integer)
    for i in l:-1:1
        setrshift!(t[i], st1)
        setcshift!(t[i], st2)
        st1 *= 2
        st2 *= 2
    end
    buildqtree!(t, l + 1)
end
setshift!(t::ShiftedQtree, l::Integer, st::Tuple{Integer, Integer}) = setshift!(t, l, st...)
setshift!(t::ShiftedQtree, st::Tuple{Integer, Integer}) = setshift!(t, 1, st)
getshift(t::ShiftedQtree, l::Integer=1) = getshift(t[l])
kernelsize(t::ShiftedQtree, l::Integer=1) = kernelsize(t[l])
getcenter(t::ShiftedQtree) = getshift(t) .+ kernelsize(t) .÷ 2
getcenter(l::Integer, a::Integer, b::Integer) = indexcenter(l, a, b)
getcenter(ind::Tuple{Integer, Integer, Integer}) = getcenter(ind...)
callefttop(t::ShiftedQtree, center) = center .- kernelsize(t) .÷  2
setcenter!(t::ShiftedQtree, center) = setshift!(t, callefttop(t, center))
function inbounds(bgqt::ShiftedQtree, qt::ShiftedQtree)
    inbounds(bgqt[1], getcenter(qt)...)
end
function outofbounds(bgqt::ShiftedQtree, qts)
    [i for (i,t) in enumerate(qts) if !inbounds(bgqt, t)]
end

################ LinkedQtree

struct QtreeNode{T}
    value::T
    children::Vector{Union{Nothing, QtreeNode}}
    parent::Union{Nothing, QtreeNode}
end

function QtreeNode{T}(value::T, parent=nothing) where T
    QtreeNode{T}(value, Vector{Union{Nothing, QtreeNode}}(nothing, 4), parent)
end
QtreeNode(value::T, parent=nothing) where T = QtreeNode{T}(value, parent)
include("qtreetools.jl")

function charmat(mat; maxlen=49)
    m,n = size(mat)
    half = maxlen ÷ 2
    maxlen = half * 2 + 1
    if m <= maxlen && n <= maxlen
        return map(x->0<x<4 ? ("▢","▩","▥")[x] : "◰", mat)
    elseif m > maxlen
        return vcat(charmat(@view(mat[1:half, :]), maxlen=maxlen),
            reshape(repeat([" ⋮"], min(maxlen, n)), 1, min(maxlen, n)),
            charmat(@view(mat[end-half+1:end, :]), maxlen=maxlen))
    else
        return hcat(charmat(@view(mat[:, 1:half]), maxlen=maxlen),
            reshape(repeat(["…"], m), m, 1),
            charmat(@view(mat[:, end-half+1:end]), maxlen=maxlen))
    end
end
        
charimage(mat; kargs...) = join([join(l)*"\n" for l in eachrow(charmat(mat; kargs...))])
charimage(qt::ShiftedQtree; maxlen=49) = charimage(qt[max(1, ceil(Int, log2(size(qt[1], 1)/maxlen))+1)], maxlen=maxlen)
Base.show(io::IO, m::MIME"text/plain", mat::PaddedMat) = print(io, "PaddedMat $(size(mat)):\n", charimage(mat))
function Base.show(io::IO, m::MIME"text/plain", qt::ShiftedQtree)
    showlevel = max(1, ceil(Int, log2(size(qt[1], 1)/49))+1)
    print(io, "ShiftedQtree{", levelnum(qt))
    if levelnum(qt)>0
        print(io, "-", size(qt[1]), "}  @level-", showlevel, ":\n", charimage(qt[showlevel], maxlen=49))
    else
        print(io, "} {}")
    end
end
end