module Stuffing
export qtree, maskqtree, qtrees, place!, overlap!, overlap, getpositions, setpositions!, packing, packing!
export QTrees, getshift, getcenter, setshift!, setcenter!, outofbounds, outofkernelbounds, 
    dynamiccollisions, partialcollisions, totalcollisions, collision, findroom_uniform, findroom_gathering,
    DynamicColliders, locate!, linked_spacial_qtree, hash_spacial_qtree
export Trainer, train!, fit!, Momentum, SGD
include("common_datatypes.jl")
include("qtrees.jl")
include("fit.jl")
using .QTrees
using .Trainer
include("utils.jl")

@debug "Threads.nthreads() = $(Threads.nthreads())"

function qtree(pic::AbstractMatrix{UInt8}, args...)
    qt = ShiftedQTree(pic, args..., default=QTrees.EMPTY) |> buildqtree!
    return qt
end
function qtree(pic::AbstractArray{Bool,2}, args...)
    pic = map(x -> ifelse(x, QTrees.FULL, QTrees.EMPTY), pic)
#     @show size(pic),m,s
    return qtree(pic, args...)
end
function qtree(pic::AbstractMatrix, args...; background=pic[1])
    qtree(pic .!= background, args...)
end

function maskqtree(pic::AbstractMatrix{UInt8})
    ms = max(size(pic)...)
    b = max(ms * 0.024, 20)
    s = 2^ceil(Int, log2(ms + b))
    qt = ShiftedQTree(pic, s, default=QTrees.FULL)
#     @show size(pic),m,s
    a, b = size(pic)
    setrshift!(qt[1], (s - a) ÷ 2)
    setcshift!(qt[1], (s - b) ÷ 2)
    return qt |> buildqtree!
end
function maskqtree(pic::AbstractArray{Bool,2})
    pic = map(x -> ifelse(x, QTrees.EMPTY, QTrees.FULL), pic)
    return maskqtree(pic)
end
function maskqtree(pic::AbstractMatrix; background=pic[1])
    maskqtree(pic .!= background)
end
# appointment: the first one is mask
function qtrees(pics; mask=nothing, background=:auto, maskbackground=:auto, size=:auto)
    ts = Vector{Stuffing.QTrees.U8SQTree}()
    if mask !== nothing
        mq = maskbackground == :auto ? maskqtree(mask) : maskqtree(mask; background=maskbackground)
        push!(ts, mq)
        size == :auto && (size = Base.size(mq[1], 1))
    elseif size == :auto
        size = maximum(maximum(Base.size(p)) for p in pics)
        size = 2^ceil(Int, log2(size))
    end
    for p in pics
        push!(ts, background == :auto ? qtree(p, size) : qtree(p, size; background=background))
    end
    ts
end
qtrees(mask, pics; kargs...) = qtrees(pics; mask=mask, kargs...)

function QTrees.place!(qtrees::AbstractVector{<:ShiftedQTree}; karg...)
    ind = QTrees.place!(deepcopy(qtrees[1]), qtrees[2:end]; karg...)
    if ind === nothing error("no room for placement") end
    qtrees
end
function QTrees.place!(qtrees::AbstractVector{<:ShiftedQTree}, inds; karg...)
    ind = QTrees.place!(deepcopy(qtrees[1]), qtrees[2:end], inds .- 1; karg...)
    if ind === nothing error("no room for placement") end
    qtrees
end
QTrees.overlap!(qtrees::AbstractVector{<:ShiftedQTree}; karg...) = QTrees.overlap!(qtrees[1], qtrees[2:end]; karg...)
QTrees.overlap(qtrees::AbstractVector{<:ShiftedQTree}; karg...) = QTrees.overlap!(deepcopy(qtrees[1]), qtrees[2:end]; karg...)

function getpositions(mask::ShiftedQTree, qtrees::AbstractVector, inds=:; mode=getshift)
    msy, msx = getshift(mask)
    pos = mode.(qtrees[inds])
    eltype(pos) <: Number && (pos = Ref(pos))
    Broadcast.broadcast(p -> (p[2] - msx + 1, p[1] - msy + 1), pos) # 左上角重合时返回(1,1)
end
function getpositions(qtrees::AbstractVector{<:ShiftedQTree}, inds=:; mode=getshift)
    @assert length(qtrees) >= 1
    getpositions(qtrees[1], @view(qtrees[2:end]), inds, mode=mode)
end
function setpositions!(mask::ShiftedQTree, qtrees::AbstractVector, inds, x_y; mode=setshift!)
    msy, msx = getshift(mask)
    eltype(x_y) <: Number && (x_y = Ref(x_y))
    Broadcast.broadcast(qtrees[inds], x_y) do qt, p
        mode(qt, (p[2] - 1 + msy, p[1] - 1 + msx))
    end
    x_y
end
function setpositions!(qtrees::AbstractVector{<:ShiftedQTree}, inds, x_y; mode=setshift!)
    @assert length(qtrees) >= 1
    setpositions!(qtrees[1], @view(qtrees[2:end]), inds, x_y, mode=mode)
end

function packing(mask, objs, args...; background=:auto, maskbackground=:auto, kargs...)
    qts = qtrees(objs, mask=mask, background=background, maskbackground=maskbackground)
    packing!(qts, args...; kargs...)
    getpositions(qts)
end
function packing!(qts, args...; kargs...)
    place!(qts)
    epochs, collisions = fit!(qts, args...; kargs...)
    @debug "$epochs epochs, $collisions collisions"
    if collisions != 0
        colllist = first.(totalcollisions(qts))
        get_text(i) = i > 1 ? "obj_$(i - 1)" : "#MASK#"
        @warn "have $(length(colllist)) collisions:\n" * 
            string([(get_text(i), get_text(j)) for (i, j) in colllist])
    end
    epochs, collisions
end
end
