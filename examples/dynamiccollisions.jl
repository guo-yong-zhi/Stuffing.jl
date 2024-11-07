using Stuffing
qts = Stuffing.randqtrees(300);
println("visualization:")
println(repr("text/plain", overlap(qts)))
function test_collision_results(C1, qts)
    C2 = QTrees.totalcollisions_native(qts);
    C3 = QTrees.totalcollisions(qts); 
    # sometimes C2!=C3. When objects are out of the scope, the `totalcollisions_native` will miss them. 
    # But `totalcollisions` may not (not promise).
    @assert length(C1) == length(C2) || length(C1) == length(C3)
    @assert Set(Set.(first.(C1))) == Set(Set.(first.(C2))) || Set(Set.(first.(C1))) == Set(Set.(first.(C3)))
end

colliders = DynamicColliders(qts);
for i in 1:10
    C1 = dynamiccollisions(colliders);
    test_collision_results(C1, qts)
    Stuffing.Trainer.batchsteps!(qts, C1) #move objects in C1
    begin #other moved labels
        QTrees.shift!(qts[3], 1, -1, -1)
        QTrees.shift!(qts[7], 1, 1, 1)
        union!(colliders, [3, 7]) #other updated labels
    end
end
