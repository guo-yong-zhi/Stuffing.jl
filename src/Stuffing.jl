module Stuffing
export qtree, maskqtree, qtrees, place!, overlap!, overlap, getpositions, setpositions!, packing, packing!
export QTrees, getshift, getcenter, setshift!, setcenter!, outofbounds, outofkernelbounds, 
    batchcollisions, collision, findroom_uniform, findroom_gathering
export Trainer, train!, fit!, Momentum
include("qtrees.jl")
include("train.jl")
using .QTrees
using .Trainer
include("utils.jl")

@info "Threads.nthreads() = $(Threads.nthreads())"

function qtree(pic::AbstractArray{Bool,2}, args...)
    pic = map(x -> ifelse(x, QTrees.FULL, QTrees.EMPTY), pic)
    qt = ShiftedQtree(pic, args..., default=QTrees.EMPTY) |> buildqtree!
#     @show size(pic),m,s
    return qt
end
function qtree(pic::AbstractMatrix, args...; background=pic[1])
    qtree(pic .!= background, args...)
end

function maskqtree(pic::AbstractArray{Bool,2})
    pic = map(x -> ifelse(x, QTrees.EMPTY, QTrees.FULL), pic)
    ms = max(size(pic)...)
    b = max(ms * 0.024, 20)
    s = 2^ceil(Int, log2(ms + b))
    qt = ShiftedQtree(pic, s, default=QTrees.FULL)
#     @show size(pic),m,s
    a, b = size(pic)
    setrshift!(qt[1], (s - a) ÷ 2)
    setcshift!(qt[1], (s - b) ÷ 2)
    return qt |> buildqtree!
end
function maskqtree(pic::AbstractMatrix; background=pic[1])
    maskqtree(pic .!= background)
end
# appointment: the first one is mask
function qtrees(pics; mask=nothing, background=:auto, maskbackground=:auto)
    ts = Vector{Stuffing.QTrees.ShiftedQtree}()
    if mask !== nothing
        mq = maskbackground == :auto ? maskqtree(mask) : maskqtree(mask; background=maskbackground)
        push!(ts, mq)
        sz = size(mq[1], 1)
    else
        sz = maximum(maximum(size(p)) for p in pics)
        sz = 2^ceil(Int, log2(sz))
    end
    for p in pics
        push!(ts, background == :auto ? qtree(p, sz) : qtree(p, sz; background=background))
    end
    ts
end
qtrees(mask, pics; kargs...) = qtrees(pics; mask=mask, kargs...)

function QTrees.place!(qtrees::AbstractVector{<:ShiftedQtree}; karg...)
    ind = QTrees.place!(deepcopy(qtrees[1]), qtrees[2:end]; karg...)
    if ind === nothing error("no room for placement") end
    qtrees
end
function QTrees.place!(qtrees::AbstractVector{<:ShiftedQtree}, inds; karg...)
    ind = QTrees.place!(deepcopy(qtrees[1]), qtrees[2:end], inds .- 1; karg...)
    if ind === nothing error("no room for placement") end
    qtrees
end
QTrees.overlap!(qtrees::AbstractVector{<:ShiftedQtree}; karg...) = QTrees.overlap!(qtrees[1], qtrees[2:end]; karg...)
QTrees.overlap(qtrees::AbstractVector{<:ShiftedQtree}; karg...) = QTrees.overlap!(deepcopy(qtrees[1]), qtrees[2:end]; karg...)

function getpositions(mask::ShiftedQtree, qtrees::AbstractVector, inds=:; type=getshift)
    msy, msx = getshift(mask)
    pos = type.(qtrees[inds])
    pos = eltype(pos) <: Number ? Ref(pos) : pos
    Broadcast.broadcast(p -> (p[2] - msx + 1, p[1] - msy + 1), pos) # 左上角重合时返回(1,1)
end
function getpositions(qtrees::AbstractVector{<:ShiftedQtree}, inds=:; type=getshift)
    @assert length(qtrees) >= 1
    getpositions(qtrees[1], @view(qtrees[2:end]), inds, type=type)
end
function setpositions!(mask::ShiftedQtree, qtrees::AbstractVector, inds, x_y; type=setshift!)
    msy, msx = getshift(mask)
    x_y = eltype(x_y) <: Number ? Ref(x_y) : x_y
    Broadcast.broadcast(qtrees[inds], x_y) do qt, p
        type(qt, (p[2] - 1 + msy, p[1] - 1 + msx))
    end
    x_y
end
function setpositions!(qtrees::AbstractVector{<:ShiftedQtree}, inds, x_y; type=setshift!)
    @assert length(qtrees) >= 1
    setpositions!(qtrees[1], @view(qtrees[2:end]), inds, x_y, type=type)
end

function packing(mask, objs, args...; background=:auto, maskbackground=:auto, kargs...)
    qts = qtrees(objs, mask=mask, background=background, maskbackground=maskbackground)
    packing!(qts, args...; kargs...)
    getpositions(qts)
end
function packing!(qts, args...; kargs...)
    place!(qts)
    ep, nc = fit!(qts, args...; kargs...)
    @info "$ep epochs, $nc collections"
    if nc != 0
        colllist = first.(batchcollisions(qts))
        get_text(i) = i > 1 ? "obj_$(i - 1)" : "#MASK#"
        @warn "have $(length(colllist)) collisions:\n" * 
            string([(get_text(i), get_text(j)) for (i, j) in colllist])
    end
    qts
end
end
