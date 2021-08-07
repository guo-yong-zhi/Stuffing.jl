using Stuffing
using Test
using Random

include("test_qtree.jl")
include("test_trainer.jl")
@testset "Stuffing.jl" begin
    qt = qtree(fill(true, 3, 4), 16)
    @test qt[1][1,1] == QTree.FULL

    qt = qtree(fill(true, 3, 4), background=true)
    @test qt[1][1,1] == QTree.EMPTY
    
    mask = fill("aa", 500, 800)
    m, n = size(mask)
    for i in 1:m
        for j in 1:n
            if (i - m / 2)^2 / (m / 2)^2 + (j - n / 2)^2 / (n / 2)^2 < 1
                mask[i,j] = "bb"
            end
        end
    end
    maskqt = maskqtree(mask, background="aa") # 椭圆空间
    @test maskqt[1][QTree.getcenter(maskqt)...] == QTree.EMPTY

    objs = [fill(true, 3, 4), fill(true, 1, 5), fill(true, 9, 9), fill(true, 12, 2)]
    qts = qtrees(objs)
    @test size(qts[1][1], 1) == size(qts[end][1], 1)
    @test size(qts[1][1], 1) >= 12

    qts = qtrees(objs, mask=mask, maskbackground="bb")
    @test size(qts[2][1], 1) >= 800
    maskqt = qts[1]
    @test maskqt[1][QTree.getcenter(maskqt)...] == QTree.FULL
    
    o1 = zeros(100, 100)
    o1[30:50, 30:50] .= 1
    o2 = zeros(100, 100)
    o2[40:80, 40:80] .= 1
    o3 = zeros(100, 100)
    o3[79:90, 79:90] .= 1
    qts = qtrees([o1, o2, o3], background=0.0)
    C = batchcollision(qts)
    @test length(C) == 2
    setshift!(qts[1], (-1000, -1000))
    @test collision(qts[1], qts[2])[1] < 0

    mask = fill(true, 500, 800) # can be any AbstractMatrix
    objs = []
    for i in 1:30
        s = 20 + randexp() * 50
        obj = fill(true, round(Int, s) + 1, round(Int, s * (0.5 + rand() / 2)) + 1) # Bool Matrix implied that background = false
        push!(objs, obj)
    end
    sort!(objs, by=prod ∘ size, rev=true)
    packing(mask, objs, 10)
    qts = qtrees(objs, mask=mask);
    setpositions!(qts, :, (200, 300))
    packing!(qts, trainer=Trainer.trainepoch_P2!)
    getpositions(qts)
    @test isempty(outofkernelbounds(qts[1], qts[2:end]))
    @test isempty(outofbounds(qts[1], qts[2:end]))
end
