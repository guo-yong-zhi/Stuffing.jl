ShiftedQtree = QTree.ShiftedQtree
buildqtree! = QTree.buildqtree!
EMPTY = QTree.EMPTY
testqtree = Stuffing.testqtree

@testset "qtree.jl" begin
    qt = ShiftedQtree(rand((0,0,1), rand(50:300), rand(50:300)))|>buildqtree!
    @test qt[1][-10,-15] == EMPTY
    @test_throws  BoundsError qt[1][-10,-15] = EMPTY
    qt2 = ShiftedQtree(rand((0,0,1), size(qt[1])))|>buildqtree!
    testqtree(qt)
    QTree.shift!(qt, 3, 2, 5)
    QTree.setshift!(qt2, 4, 1, 2)
    testqtree(qt)
    testqtree(qt2)
    QTree.overlap!(qt2, qt)
    testqtree(qt2)
    qt = ShiftedQtree(rand((0,0,1), 1, 1))|>buildqtree!
    @test QTree.levelnum(qt) == 1
    qt = ShiftedQtree(rand((0,0,1), 1, 2))|>buildqtree!
    @test QTree.levelnum(qt) == 2
    @test_throws  AssertionError qt = ShiftedQtree(rand((0,0,1), 0, 0))|>buildqtree!

    qt = ShiftedQtree(rand((0,0,0,1), rand(50:300), rand(50:300)), 512)|>buildqtree!
    li = QTree.locate(qt)
    @test qt[li]!=QTree.EMPTY
    for l in QTree.levelnum(qt)
        if l >= li[1]
            @test sum(qt[li[1]].!=QTree.EMPTY) <= 1
        else
            @test sum(qt[li[1]-1].!=QTree.EMPTY) > 1
        end
    end
    
    objs = []
    for i in 1:30
        s = 20 + randexp() * 60
        obj = fill(true, round(Int, s)+1, round(Int, s*(0.5+rand()/2))+1)
        push!(objs, obj)
    end
    mask = fill(true, 500, 500)
    qts = qtrees(mask, objs)
    placement!(qts, roomfinder=QTree.findroom_gathering)
    placement!(qts)
    clq = QTree.batchcollision_qtree(qts)
    cln = QTree.batchcollision_native(qts)
    @test Set(Set.(first.(clq))) == Set(Set.(first.(cln)))
end