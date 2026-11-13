using XDiag
using LinearAlgebra

function main(B_T::Real = 0.0)
    N = 3
    J_K  = 5.0            # Trif's J
    D_K  = -1 / sqrt(3)   # same number as your benchmark's |DM|
    gfac = 4 / 3          # K/T, same factor as B_code = (4/3) B_T

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
    H["B"]  = gfac * B_T
    H["Dz"] = D_K

    set_verbosity(2)          # set verbosity for monitoring progress
    e0 = eigval0(H, block)    # compute ground state energy
    offset = 3.75
    difference = e0+offset

    println("Ground state energy (B_T = $B_T): $difference")
    return difference 
end
