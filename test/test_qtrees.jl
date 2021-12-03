ShiftedQTree = QTrees.ShiftedQTree
buildqtree! = QTrees.buildqtree!
EMPTY = QTrees.EMPTY
testqtree = Stuffing.testqtree

@testset "qtrees.jl" begin
    qt = ShiftedQTree(rand((0, 0, 1), rand(50:300), rand(50:300))) |> buildqtree!
    @test qt[1][-10,-15] == EMPTY
    @test_throws  BoundsError qt[1][-10,-15] = EMPTY
    qt2 = ShiftedQTree(rand((0, 0, 1), size(qt[1]))) |> buildqtree!
    testqtree(qt)
    QTrees.shift!(qt, 3, 2, 5)
    QTrees.setshift!(qt2, 4, 1, 2)
    testqtree(qt)
    testqtree(qt2)
    QTrees.overlap!(qt2, qt)
    testqtree(qt2)
    qt = ShiftedQTree(rand((0, 0, 1), 1, 1)) |> buildqtree!
    @test QTrees.levelnum(qt) == 1
    qt = ShiftedQTree(rand((0, 0, 1), 1, 2)) |> buildqtree!
    @test QTrees.levelnum(qt) == 2
    @test_throws  AssertionError qt = ShiftedQTree(rand((0, 0, 1), 0, 0)) |> buildqtree!

    qt = ShiftedQTree(rand((0, 0, 0, 1), rand(50:300), rand(50:300)), 512) |> buildqtree!
    li = QTrees.locate(qt)
    @test qt[li] != QTrees.EMPTY
    for l in QTrees.levelnum(qt)
        if l >= li[1]
            @test sum(qt[li[1]] .!= QTrees.EMPTY) <= 1
        else
            @test sum(qt[li[1] - 1] .!= QTrees.EMPTY) > 1
        end
    end
    
    objs = []
    for i in 1:15
        s = 20 + randexp() * 40
        obj = fill(true, round(Int, s) + 1, round(Int, s * (0.5 + rand() / 2)) + 1)
        push!(objs, obj)
    end
    mask = fill(true, 500, 500)
    qts = qtrees(mask, objs)
    place!(qts, roomfinder=QTrees.findroom_gathering)
    place!(qts)
    clq = QTrees.batchcollisions_region(qts)
    cln = QTrees.batchcollisions_native(qts)
    @test Set(Set.(first.(clq))) == Set(Set.(first.(cln)))
end