using Stuffing
o1 = zeros(100, 100);
o1[35:50, 30:55] .= 1;
o2 = zeros(100, 100);
o2[45:80, 50:80] .= 1;
o3 = zeros(100, 100);
o3[60:90, 67:90] .= 1;

qts = qtrees([o1, o2, o3], background=0.0);
println("visualization:")
println(repr("text/plain", overlap(qts)))

println("# collision")
println("If the first number is positive，it means tow objects collided, and the value can represent the severity in some sense. ",
"The last two numbers are coordinates where the collision occurred.")
C = [(i, j) => collision(qts[i], qts[j]) for (i, j) in [(1, 2),(1, 3),(2, 3)]]
println(C)

println("# totalcollisions")
C = totalcollisions(qts)
println(C)

for (k, p) in C
    println(k, " collided at ", QTrees.indexrange(p))
end

println("# partialcollisions")
qts = Stuffing.randqtrees(300);
println("visualization:")
println(repr("text/plain", overlap(qts)))
labels = [1, 3, 99]
spqtree = QTrees.locate!(qts, linked_spacial_qtree(qts));
C1 = partialcollisions(qts, spqtree, Set(labels));
@show length(C1)
println("Outputs the collisions associated with $labels.")
println(C1)
C2 = totalcollisions(qts);
@show length(C2)
@assert Set([Set(p) for p in first.(C2) if !isdisjoint(p, labels)]) == Set(Set.(first.(C1)))
