using Stuffing
using Test
using Random

include("test_qtree.jl")
include("test_trainer.jl")
@testset "Stuffing.jl" begin
    qt = qtree(fill(true,3,4), 16)
    @test qt[1][1,1] == QTree.FULL

    qt = qtree(fill(true,3,4), background=true)
    @test qt[1][1,1] == QTree.EMPTY
    
    mask = fill("aa", 500, 800)
    m,n = size(mask)
    for i in 1:m
        for j in 1:n
            if (i-m/2)^2/(m/2)^2 + (j-n/2)^2/(n/2)^2 < 1
                mask[i,j] = "bb"
            end
        end
    end
    maskqt = maskqtree(mask, background="aa") #椭圆空间
    @test maskqt[1][QTree.getcenter(maskqt)...] == QTree.EMPTY

    objs = [fill(true,3,4), fill(true,1,5), fill(true,9,9), fill(true,12,2)]
    qts = qtrees(objs)
    @test size(qts[1][1], 1) == size(qts[end][1], 1)
    @test size(qts[1][1], 1) >= 12

    qts = qtrees(objs, mask=mask, maskbackground="bb")
    @test size(qts[2][1], 1) >= 800
    maskqt = qts[1]
    @test maskqt[1][QTree.getcenter(maskqt)...] == QTree.FULL
end
