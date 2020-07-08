@info("BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN")
println("BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN")

function all_lteq(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    all(x[1] <= x[2] <= x[3] for x in zip(A, B, C))
end

A = Float32[-1.0, -1.0]
B = [0.0, 0.0]
C = Float32[1.0, 1.0]
all_lteq(A, B, C)

@info("DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE")
println("DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE")
