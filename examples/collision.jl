using Stuffing
o1 = zeros(100, 100)
o1[35:50, 30:55] .= 1
o2 = zeros(100, 100)
o2[45:80, 50:80] .= 1
o3 = zeros(100, 100)
o3[60:90, 67:90] .= 1

qts = qtrees([o1, o2, o3], background=0.0)
println("The first number is positive means tow objs collided, and the value can represent the severity in some sense. ",
"The last two numbers are coordinates.")
C = [(i, j) => collision(qts[i], qts[j]) for (i, j) in [(1, 2),(1, 3),(2, 3)]]
println(C)
# or
C = batchcollision(qts)
println(C)

for (k, p) in C
    println(k, " collided at ", QTree.indexrange(p))
end
println("visualization:")
println(repr("text/plain", overlap(qts)))