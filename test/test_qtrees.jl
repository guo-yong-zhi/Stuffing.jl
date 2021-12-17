ShiftedQTree = QTrees.ShiftedQTree
buildqtree! = QTrees.buildqtree!
testqtree = Stuffing.testqtree

@testset "qtrees.jl" begin
    qt = ShiftedQTree(rand((0, 0, 1), rand(50:300), rand(50:300))) |> buildqtree!
    @test qt[1][-10, -15] == QTrees.EMPTY
    @test_throws BoundsError qt[1][-10, -15] = QTrees.EMPTY
    qt2 = ShiftedQTree(rand((0, 0, 1), size(qt[1]))) |> buildqtree!
    testqtree(qt)
    QTrees.shift!(qt, 3, 2, 5)
    QTrees.setshift!(qt2, 4, 1, 2)
    testqtree(qt)
    testqtree(qt2)
    QTrees.overlap!(qt2, qt)
    testqtree(qt2)
    qt = ShiftedQTree(rand((0, 0, 1), 1, 1)) |> buildqtree!
    @test QTrees.length(qt) == 2
    qt = ShiftedQTree(rand((0, 0, 1), 1, 2)) |> buildqtree!
    @test QTrees.length(qt) == 2
    @test_throws AssertionError qt = ShiftedQTree(rand((0, 0, 1), 0, 0)) |> buildqtree!

    qt = ShiftedQTree(rand((0, 0, 0, 1), rand(50:300), rand(50:300)), 512) |> buildqtree!

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

    #corner cases
    # batch
    @test QTrees.batchcollisions_region(Vector{QTrees.U8SQTree}()) |> isempty
    @test QTrees.batchcollisions_native(Vector{QTrees.U8SQTree}()) |> isempty
    qt1 = qtree(reshape([1], 1, 1))
    @test QTrees.batchcollisions_region([qt1]) |> isempty
    @test QTrees.batchcollisions_native([qt1]) |> isempty
    # empty tree
    qt1 = qtree(Matrix{Int}(undef, 0, 0), background = 0)
    qt2 = qtree(Matrix{Int}(undef, 0, 0), background = 0)
    @test length(qt1) >= 2
    @test QTrees.collision_dfs(qt1, qt2)[1] < 0
    @test QTrees.collision(qt1, qt2)[1] < 0
    @test QTrees.batchcollisions_native([qt1, qt2]) |> isempty
    @test QTrees.batchcollisions_region([qt1, qt2]) |> isempty
    # no pixel tree
    qt1 = qtree(reshape([1], 1, 1))
    qt2 = qtree(reshape([1], 1, 1))
    @test length(qt1) >= 2
    @test QTrees.collision_dfs(qt1, qt2)[1] < 0
    @test QTrees.collision(qt1, qt2)[1] < 0
    @test QTrees.batchcollisions_native([qt1, qt2]) |> isempty
    @test QTrees.batchcollisions_region([qt1, qt2]) |> isempty
    # single pixel tree
    qt1 = qtree(reshape([1], 1, 1), background=0)
    qt2 = qtree(reshape([1], 1, 1), background=0)
    @test length(qt1) >= 2
    @test QTrees.collision_dfs(qt1, qt2) == (1, 1, 1)
    @test QTrees.collision(qt1, qt2) == (1, 1, 1)
    @test QTrees.batchcollisions_native([qt1, qt2]) |> only |> last == (1, 1, 1)
    @test QTrees.batchcollisions_region([qt1, qt2]) |> only |> last == (1, 1, 1)
    # out of bounds tree
    setshift!(qt1, (10, -10))
    @test QTrees.collision_dfs(qt1, qt2)[1] < 0
    @test QTrees.collision(qt1, qt2)[1] < 0
    @test QTrees.batchcollisions_native([qt1, qt2]) |> isempty
    @test QTrees.batchcollisions_region([qt1, qt2]) |> isempty
end