using XDiag
using LinearAlgebra

function main(N::Int, B::Real = 0.0, B_is_tesla::Bool = true)
    J_K  = 5.0            # Trif's J
    D_K  = -1 / sqrt(3)   # same number as your benchmark's |DM|
    gfac = 4 / 3          # K/T, same factor as B_code = (4/3) B[T]

    block = Spinhalf(N)

    sx = [0 0.5; 0.5 0]
    sy = [0 -0.5im; 0.5im 0]

    dm = kron(sx, sy) - kron(sy, sx)   # Sx⊗Sy − Sy⊗Sx, complex matrix

    H = OpSum()
    for i in 1:N
        j = mod1(i + 1, N)
        H += "J" * Op("SdotS", [i, j])
        H += "B" * Op("Sz", i)
        H += "Dz" * Op("Matrix", [i, j], dm)
    end
    H["J"]  = J_K
    H["B"]  = B * (B_is_tesla ? gfac : 1)
    H["Dz"] = D_K

    set_verbosity(2)          # set verbosity for monitoring progress
    e0 = eigval0(H, block)    # compute ground state energy
    # offset = 3.75           # offset for N=3 compare to trif graphs
    # difference = e0+offset

    println("Ground state energy (B[$(B_is_tesla ? "T" : "k")] = $B): $e0")
    return e0
end

function Run(
    N::Int = 5,
    B_vals::AbstractVector{<:Real} = [0.0, 0.25, 0.5],
    B_is_tesla::Bool = false
)
    results = [main(N, b, B_is_tesla) for b in B_vals]

    unit = B_is_tesla ? "T" : "k"
    b_header = "B [$unit]"

    println("\n+-----+----------+----------------------+")
    println("|  N  | $(rpad(b_header, 8)) | Energy (e0 + offset) |")
    println("+-----+----------+----------------------+")
    for (b, val) in zip(B_vals, results)
        println("| $(lpad(N, 3)) | $(lpad(b, 8)) | $(rpad(round(val, digits=10), 20)) |")
    end
    println("+-----+----------+----------------------+")

    return results
end

# gives all eigenvalues for a given state rather than just minimum
function all_vals(N::Int = 5, B::Real = 0.0, B_is_tesla::Bool = false)
    J_K  = 5.0            # Trif's J
    D_K  = -1 / sqrt(3)   # same number as your benchmark's |DM|
    gfac = 4 / 3          # K/T, same factor as B_code = (4/3) B[T]

    block = Spinhalf(N)

    sx = [0 0.5; 0.5 0]
    sy = [0 -0.5im; 0.5im 0]

    dm = kron(sx, sy) - kron(sy, sx)   # Sx⊗Sy − Sy⊗Sx, complex matrix

    H = OpSum()
    for i in 1:N
        j = mod1(i + 1, N)
        H += "J" * Op("SdotS", [i, j])
        H += "B" * Op("Sz", i)
        H += "Dz" * Op("Matrix", [i, j], dm)
    end
    H["J"]  = J_K
    H["B"]  = B * (B_is_tesla ? gfac : 1)
    H["Dz"] = D_K

    set_verbosity(2)          # set verbosity for monitoring progress
    vals = eigvals_lanczos(H, block, neigvals=N)
    @show vals.eigenvalues

end
