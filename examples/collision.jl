using Stuffing

o1 = zeros(100, 100)
o1[30:50, 30:50] .= 1
o2 = zeros(100, 100)
o2[40:80, 40:80] .= 1
o3 = zeros(100, 100)
o3[79:90, 79:90] .= 1

qts = qtrees([o1, o2, o3], background=0.0)
C = [collision_bfs_rand(qts[i], qts[j]) for (i, j) in [(1,2),(1,3),(2,3)]] #positive means have collision
println(C)
# or
C = batchcollision(qts)
println(C)

for (k, p) in C
    println(k, " collided at ", QTree.indexrange(p))
end