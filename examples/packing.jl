using Stuffing
using Random

# prepare some pseudo images 
mask = fill("aa", 500, 800) #can be any AbstractMatrix
m,n = size(mask)
for i in 1:m
    for j in 1:n
        if (i-m/2)^2/(m/2)^2 + (j-n/2)^2/(n/2)^2 < 1
            mask[i,j] = "bb"
        end
    end
end
objs = []
for i in 1:30
    s = 20 + randexp() * 60
    obj = fill(true, round(Int, s)+1, round(Int, s*(0.5+rand()/2))+1) #Bool Matrix implied that background = false
    push!(objs, obj)
end
sort!(objs, by=prod∘size, rev=true) #better in descending order of size

#packing
qts = qtrees(objs, mask=mask, maskbackground="aa")
place!(qts)
fit!(qts)

# draw
println("visualization:")
oqt = overlap(qts)
println(repr("text/plain",oqt))
#or
println(QTree.charimage(oqt, maxlen=97))
#or
using Colors
imageof(qt) = Gray.(QTree.decode.(qt[1]))
imageof(oqt)

#get layout
println("layout:")
positions = getpositions(qts)
println(positions)

println("or get layout directly")
packing(mask, objs, maskbackground="aa")
#or
qts = qtrees(objs, mask=mask, maskbackground="aa");
packing!(qts)
getpositions(qts)
