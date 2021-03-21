module Stuffing
export qtree, maskqtree, qtrees, placement!, overlap!, overlap, getpositions, setpositions!, packing, packing!
export QTree, getshift, getcenter, setshift!, setcenter!, outofbounds, batchcollision, 
collision_bfs_rand, findroom_uniform, findroom_gathering
export Trainer, train!, fit!, Momentum
# Write your package code here.
include("qtree.jl")
include("train.jl")
using .QTree
using .Trainer
include("utils.jl")

function qtree(pic::AbstractArray{Bool,2}, args...)
    pic = map(x -> ifelse(x, QTree.FULL, QTree.EMPTY), pic)
    qt = ShiftedQtree(pic, args..., default=QTree.EMPTY) |> buildqtree!
#     @show size(pic),m,s
    return qt
end
function qtree(pic::AbstractMatrix, args...; background=pic[1])
    qtree(pic .!= background, args...)
end

function maskqtree(pic::AbstractArray{Bool,2})
    pic = map(x -> ifelse(x, QTree.EMPTY, QTree.FULL), pic)
    ms = max(size(pic)...)
    b = max(ms * 0.024, 20)
    s = 2^ceil(Int, log2(ms+b))
    qt = ShiftedQtree(pic, s, default=QTree.FULL)
#     @show size(pic),m,s
    a, b = size(pic)
    setrshift!(qt[1], (s-a)÷2)
    setcshift!(qt[1], (s-b)÷2)
    return qt |> buildqtree!
end
function maskqtree(pic::AbstractMatrix; background=pic[1])
    maskqtree(pic .!= background)
end

function qtrees(pics; mask=nothing, background=:auto, maskbackground=:auto)
    ts = Vector{Stuffing.QTree.ShiftedQtree}()
    if mask !== nothing
        mq = maskbackground==:auto ? maskqtree(mask) : maskqtree(mask; background=maskbackground)
        push!(ts, mq)
        sz = size(mq[1], 1)
    else
        sz = maximum(maximum(size(p)) for p in pics)
        sz = 2^ceil(Int, log2(sz))
    end
    for p in pics
        push!(ts, background==:auto ? qtree(p, sz) : qtree(p, sz; background=background))
    end
    ts
end
qtrees(mask, pics; kargs...) = qtrees(pics; mask=mask, kargs...)
#??
function QTree.placement!(qtrees::AbstractVector{<:ShiftedQtree}; karg...)
    ind = QTree.placement!(deepcopy(qtrees[1]), qtrees[2:end]; karg...)
    if ind === nothing error("no room for placement") end
    qtrees
end
function QTree.placement!(qtrees::AbstractVector{<:ShiftedQtree}, inds; karg...)
    ind = QTree.placement!(deepcopy(qtrees[1]), qtrees[2:end], inds.-1; karg...)
    if ind === nothing error("no room for placement") end
    qtrees
end
QTree.overlap!(qtrees::AbstractVector{<:ShiftedQtree}; karg...) = QTree.overlap!(qtrees[1], qtrees[2:end]; karg...)
QTree.overlap(qtrees::AbstractVector{<:ShiftedQtree}; karg...) = QTree.overlap!(deepcopy(qtrees[1]), qtrees[2:end]; karg...)

function getpositions(qts; type=getshift)
    mqt = qts[1]
    msy, msx = getshift(mqt)
    pos = getshift.(qts[2:end])
    Broadcast.broadcast(p->(p[2]-msx+1, p[1]-msy+1), pos) #左上角重合时返回(1,1)
end

function setpositions!(qts, x_y; type=setshift!)
    mqt = qts[1]
    msy, msx = getshift(mqt)
    x_y = eltype(x_y) <: Number ? Ref(x_y) : x_y
        Broadcast.broadcast(qts[2:end], x_y) do qt, p
        type(qt, (p[2]-1+msy, p[1]-1+msx))
    end
    x_y
end
function packing(mask, objs; background=:auto, maskbackground=:auto)
    qts = qtrees(objs, mask=mask, background=background, maskbackground=maskbackground) |> packing! 
    getpositions(qts)
end
function packing!(qts)
    placement!(qts)
    ep, nc = fit!(qts)
    println("$ep epochs, $nc collections")
    if nc != 0
        colllist = first.(batchcollision(qts))
        println("have $(length(colllist)) collisions:")
        get_text(i) = i>1 ? "obj$(i-1)" : "#MASK#"
        println([(get_text(i), get_text(j)) for (i,j) in colllist])
    end
    qts
end
end
