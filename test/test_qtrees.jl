ShiftedQTree = QTrees.ShiftedQTree
buildqtree! = QTrees.buildqtree!
testqtree = Stuffing.testqtree

@testset "qtrees.jl" begin
    ind = (rand(1:20), rand(1:2000), rand(1:2000))
    cn = QTrees.childnumber(ind)
    @test cn == QTrees.childnumber(QTrees.parent(ind), ind)
    @test QTrees.child(QTrees.parent(ind), cn) == ind

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

    qts = Stuffing.randqtrees(200)
    #locate!
    spt2 = QTrees.locate!(qts, QTrees.linked_spacial_qtree(qts))
    spt1 = QTrees.locate!(qts, QTrees.hash_spacial_qtree())
    spt1_ = Dict([p for p in QTrees.tree(spt1) if QTrees.inrange(QTrees.spacial_index(QTrees.tree(spt2)), first(p))]) #inrange filter
    spt2_ = QTrees.collect_tree(spt2) #to hash
    @test Dict([k=>Set(v) for (k,v) in spt1_]) == Dict([k=>Set(v) for (k,v) in spt2_])
    empty!(spt2, 1)
    spt2_ = QTrees.collect_tree(spt2)
    for inds2 in values(spt2_)
        @assert !(1 in inds2)
    end

    #collisions
    cls = QTrees.totalcollisions_spacial(qts)
    cln = QTrees.totalcollisions_native(qts)
    clp = QTrees.partialcollisions(qts, spt2, Set(1:length(qts)))
    @test Set(Set.(first.(cls))) == Set(Set.(first.(cln))) == Set(Set.(first.(clp)))
    updated = first.(clp)|>Iterators.flatten|>Set
    clp = QTrees.partialcollisions(qts, spt2, updated, unique=true)
    @test Set(Set.(first.(cln))) == Set(Set.(first.(clp)))
    clp = QTrees.partialcollisions(qts)
    @test Set(Set.(first.(cln))) == Set(Set.(first.(clp)))
    # partial
    labels = [1,3,6,99]
    sptree = QTrees.locate!(qts, linked_spacial_qtree(qts))
    c1 = QTrees.partialcollisions(qts, sptree, Set(labels))
    c2set = Set([Set(p) for p in first.(QTrees.totalcollisions(qts)) if !isdisjoint(p, labels)])
    @test c2set == Set(Set.(first.(c1)))

    #edge cases
    # batch
    @test QTrees.totalcollisions_spacial(Vector{QTrees.U8SQTree}()) |> isempty
    @test QTrees.totalcollisions_native(Vector{QTrees.U8SQTree}()) |> isempty
    qt1 = qtree(reshape([1], 1, 1))
    @test QTrees.totalcollisions_spacial([qt1]) |> isempty
    @test QTrees.totalcollisions_native([qt1]) |> isempty
    @test QTrees.partialcollisions([qt1]) |> isempty
    # empty tree
    qt1 = qtree(Matrix{Int}(undef, 0, 0), background = 0)
    qt2 = qtree(Matrix{Int}(undef, 0, 0), background = 0)
    @test length(qt1) >= 2
    @test QTrees.collision_dfs(qt1, qt2)[1] < 0
    @test QTrees.collision(qt1, qt2)[1] < 0
    @test QTrees.totalcollisions_native([qt1, qt2]) |> isempty
    @test QTrees.totalcollisions_spacial([qt1, qt2]) |> isempty
    @test QTrees.partialcollisions([qt1, qt2]) |> isempty
    # no pixel tree
    qt1 = qtree(reshape([1], 1, 1))
    qt2 = qtree(reshape([1], 1, 1))
    @test length(qt1) >= 2
    @test QTrees.collision_dfs(qt1, qt2)[1] < 0
    @test QTrees.collision(qt1, qt2)[1] < 0
    @test QTrees.totalcollisions_native([qt1, qt2]) |> isempty
    @test QTrees.totalcollisions_spacial([qt1, qt2]) |> isempty
    @test QTrees.partialcollisions([qt1, qt2]) |> isempty
    # single pixel tree
    qt1 = qtree(reshape([1], 1, 1), background=0)
    qt2 = qtree(reshape([1], 1, 1), background=0)
    @test length(qt1) >= 2
    @test QTrees.collision_dfs(qt1, qt2) == (1, 1, 1)
    @test QTrees.collision(qt1, qt2) == (1, 1, 1)
    @test QTrees.totalcollisions_native([qt1, qt2]) |> only |> last == (1, 1, 1)
    @test QTrees.totalcollisions_spacial([qt1, qt2]) |> only |> last == (1, 1, 1)
    @test QTrees.partialcollisions([qt1, qt2]) |> only |> last == (1, 1, 1)
    spt2 = QTrees.locate!([qt1, qt2], QTrees.linked_spacial_qtree([qt1, qt2]))
    @test QTrees.partialcollisions([qt1, qt2], spt2, Set([1])) |> only |> last == (1, 1, 1)
    # out of bounds tree
    setshift!(qt1, (10, -10))
    @test QTrees.collision_dfs(qt1, qt2)[1] < 0
    @test QTrees.collision(qt1, qt2)[1] < 0
    @test QTrees.totalcollisions_native([qt1, qt2]) |> isempty
    @test QTrees.totalcollisions_spacial([qt1, qt2]) |> isempty
    @test QTrees.partialcollisions([qt1, qt2]) |> isempty
end