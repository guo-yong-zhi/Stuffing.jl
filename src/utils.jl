import .QTrees.decode
bitor(l) = reduce((a, b) -> a .| b, l)

FULL = QTrees.FULL
EMPTY = QTrees.EMPTY
MIX = QTrees.MIX
function testqtree(qt)
    for l in 2:QTrees.length(qt)
        for i in 1:size(qt[l], 1)
            for j in 1:size(qt[l], 2)
                c = [qt[l - 1, 2i, 2j], qt[l - 1, 2i - 1, 2j], qt[l - 1, 2i, 2j - 1], qt[l - 1, 2i - 1, 2j - 1]]
                if qt[l, i, j] == FULL
                    @assert all(c .== FULL) (qt[l, i, j], (l, i, j))
                elseif qt[l, i, j] == EMPTY
                    @assert all(c .== EMPTY) (qt[l, i, j], (l, i, j))
                elseif qt[l, i, j] == MIX
                    @assert !(all(c .== FULL) || all(c .== EMPTY)) (qt[l, i, j], (l, i, j))
                else
                    error(qt[l, i, j], (l, i, j))
                end
            end
        end
    end
end

function randobject(sz=100)
    s1 = max(2, sz/10) + 0.9rand() * sz
    s2 = max(2, sz/10) + 0.9rand() * sz
    trues(round(Int, s1), round(Int, s2))
end
function randobjects(n, sz=100)
    objs = [randobject(sz) for i in 1:n]
    sort!(objs, by=prod ∘ size, rev=true)
end
function randqtrees(n; sz=100, masksize=(1000, 1000))
    objs = randobjects(n, sz)
    qts = qtrees(objs, mask=trues(masksize...))
    place!(qts)
end