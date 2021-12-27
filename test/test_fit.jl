@testset "fit.jl" begin
    x = rand(Float64, 1000) * 1000;
    @test sum(floor.(Int, log2.(x))) == sum(Trainer.intlog2.(x))

    lru = Trainer.LRU{Int}()
    for i in 1:10
        push!(lru, i)
    end
    for i in 10:-2:2
        push!(lru, i)
    end
    push!(lru, 1)
    @test collect(lru) == [1,2,4,6,8,10,9,7,5,3]
    lru = Trainer.intlru(10)
    for i in 1:10
        push!(lru, i)
    end
    for i in 10:-2:2
        push!(lru, i)
    end
    push!(lru, 1)
    @test collect(lru) == [1,2,4,6,8,10,9,7,5,3]
    @test collect(lru, 3) == [1,2,4]
    push!.(lru, 7:9)
    @test collect(lru, 3) == [9,8,7]
    using Random
    Random.seed!(9)
    mask = fill("aa", 500, 800) # can be any AbstractMatrix
    m, n = size(mask)
    for i in 1:m
        for j in 1:n
            if (i - m / 2)^2 / (m / 2)^2 + (j - n / 2)^2 / (n / 2)^2 < 1
                mask[i,j] = "bb"
            end
        end
    end
    objs = []
    for i in 1:100
        s = 50
        obj = fill(true, round(Int, s) + 1, round(Int, s * (0.5 + rand() / 2)) + 1) # Bool Matrix implied that background is `false`
        push!(objs, obj)
    end
    sort!(objs, by=prod ∘ size, rev=true)
    ts = [Trainer.trainepoch_E!,Trainer.trainepoch_EM!,
    Trainer.trainepoch_EM2!,Trainer.trainepoch_EM3!,
    Trainer.trainepoch_P!,Trainer.trainepoch_P2!,Trainer.trainepoch_Px!,
    Trainer.trainepoch_D!, ]
    qts = qtrees(objs, mask=mask, maskbackground="aa")

    #fit!
    setshift!(qts[2], 1, 1000, 1000);
    @test !isempty(QTrees.totalcollisions_spacial(qts[1:2]))
    l = length(qts) - 1
    repo = [0.3, true, false, i -> i % 2 == 1, round(Int, l * 0.2), [10:l...]]
    @time for t in ts
        for rp in repo
            place!(qts)
            @time fit!(qts, trainer=t, reposition=rp)
        end
    end
    place!(qts)
    fit!(qts, 100, optimiser=SGD(), patient=5)
    place!(qts)
    fit!(qts, 100, optimiser=Momentum(), patient=5)
end