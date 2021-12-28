using Stuffing
qts = Stuffing.randqtrees(300);
println("visualization:")
println(repr("text/plain", overlap(qts)))

println("""To get all collisions via `partialcollisions`, the set `updated` should contains all collided labels in last round.
        And the labels of other updated objects should be contained too.
        `partialcollisions` is faster than `totalcollisions` when `updated` is small.
        """)
updated = Set(1:length(qts));
sptree = linked_spacial_qtree(qts);
for i in 1:10
    C1 = partialcollisions(qts, sptree, updated);
    union!(updated, first.(C1) |> Iterators.flatten) #all collided labels
    C2 = QTrees.totalcollisions_native(qts);
    @assert length(C1) == length(C2)
    @assert Set(Set.(first.(C1))) == Set(Set.(first.(C2)))
    Stuffing.Trainer.batchsteps!(qts, C1)
    QTrees.shift!(qts[3], 1, 1, 1)
    QTrees.shift!(qts[7], 1, -1, -1)
    union!(updated, [3, 7]) #other updated labels
end
println("And things are similar for `dynamiccollisions`. ")
println("`dynamiccollisions` is faster than `partialcollisions` when `updated` is not that small.")
updated = UpdatedSet(length(qts));
sptree = linked_spacial_qtree(qts);
for i in 1:10
    C1 = dynamiccollisions(qts, sptree, updated);
    union!(updated, first.(C1) |> Iterators.flatten) #all collided labels
    C2 = QTrees.totalcollisions_native(qts);
    C3 = QTrees.totalcollisions(qts); 
    #sometimes C2!=C3. When objects are out of the scope, the `totalcollisions_native` will miss them. 
    #But `totalcollisions` may not (not promise).
    @assert length(C1) == length(C2) || length(C1) == length(C3)
    @assert Set(Set.(first.(C1))) == Set(Set.(first.(C2))) || Set(Set.(first.(C1))) == Set(Set.(first.(C3)))
    Stuffing.Trainer.batchsteps!(qts, C1)
    QTrees.shift!(qts[3], 1, -1, -1)
    QTrees.shift!(qts[7], 1, 1, 1)
    union!(updated, [3, 7]) #other updated labels
end