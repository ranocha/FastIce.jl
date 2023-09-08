using KernelAbstractions
using MPI
using AMDGPU

# using CairoMakie

include("mpi_utils.jl")
include("mpi_utils2.jl")

@kernel function diffusion_kernel!(A_new, A, h, _dx, _dy, _dz, offset)
    ix, iy, iz = @index(Global, NTuple)
    ix += offset[1] - 1
    iy += offset[2] - 1
    iz += offset[3] - 1
    if ix ∈ axes(A_new, 1)[2:end-1] && iy ∈ axes(A_new, 2)[2:end-1] && iz ∈ axes(A_new, 3)[2:end-1]
        @inbounds A_new[ix, iy, iz] = A[ix, iy, iz] + h * ((A[ix-1, iy  , iz  ] + A[ix+1, iy  , iz  ] - 2.0 * A[ix, iy, iz]) * _dx * _dx +
                                                           (A[ix  , iy-1, iz  ] + A[ix  , iy+1, iz  ] - 2.0 * A[ix, iy, iz]) * _dy * _dy +
                                                           (A[ix  , iy  , iz-1] + A[ix  , iy  , iz+1] - 2.0 * A[ix, iy, iz]) * _dz * _dz  )
    end
end

function main(backend=CPU(), T::DataType=Float64, dims=(0, 0, 0))
    # physics
    l = 10.0
    # numerics
    nt = 10
    nx, ny, nz = 1024, 1024, 1024
    b_width = (16, 8, 4)
    dims, comm, me, neighbors, coords, dev_id = init_distributed(dims; init_MPI=true)
    dx, dy, dz = l ./ (nx, ny, nz)
    _dx, _dy, _dz = 1.0 ./ (dx, dy, dz)
    h = min(dx, dy ,dz)^2 / 6.1
    # init arrays
    x_g = (ix, dx) -> (coords[1] * (nx - 2) + (ix-1)) * dx + dx/2
    y_g = (iy, dy) -> (coords[2] * (ny - 2) + (iy-1)) * dy + dy/2
    z_g = (iz, dz) -> (coords[3] * (nz - 2) + (iz-1)) * dz + dz/2
    # Gaussian
    # A_ini = zeros(T, nx, ny, nz)
    # A_ini .= [exp(-(x_g(ix, dx) - l/2)^2 - (y_g(iy, dy) - l/2)^2 - (z_g(iz, dz) - l/2)^2) for ix=1:size(A_ini, 1), iy=1:size(A_ini, 2), iz=1:size(A_ini, 3)]
    A_ini = rand(T, nx, ny, nz)

    A     = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    A_new = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    KernelAbstractions.copyto!(backend, A, A_ini)
    KernelAbstractions.synchronize(backend)
    A_new = copy(A)

    ### to be hidden later
    ranges = split_ndrange(A, b_width)

    exchangers = ntuple(Val(length(neighbors))) do dim
        ntuple(2) do side
            rank   = neighbors[dim][side]
            halo   = get_recv_view(Val(side), Val(dim), A_new)
            border = get_send_view(Val(side), Val(dim), A_new)
            range  = ranges[2*(dim-1) + side]
            offset, ndrange = first(range), size(range)
            Exchanger(backend, comm, rank, dev_id, halo, border) do compute_bc
                diffusion_kernel!(backend, 256)(A_new, A, h, _dx, _dy, _dz, offset; ndrange)
                if compute_bc
                    # apply_bcs!(Val(dim), fields, bcs.velocity)
                end
                KernelAbstractions.synchronize(backend)
            end
        end
    end
    ### to be hidden later

    # actions
    for it = 1:nt
        # copyto!(A, A_new)
        diffusion_kernel!(backend, 256)(A_new, A, h, _dx, _dy, _dz, first(ranges[end]); ndrange=size(ranges[end]))
        for dim in reverse(eachindex(neighbors))
            notify.(exchangers[dim])
            wait.(exchangers[dim])
        end
        KernelAbstractions.synchronize(backend)
    end

    # for dim in eachindex(neighbors)
    #     setdone!.(exchangers[dim])
    # end

    # A_g = (me==0) ? allocate(CPU(), dtype, nx_g, ny_g) : nothing
    # A_i = Array(A_new[2:end-1, 2:end-1, 2:end-1])
    # gather!(A_g, A_i, comm)

    # @info "visu"
    # fig = Figure(resolution=(1000,1000), fontsize=32)
    # ax  = Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="y")
    # plt = heatmap!(ax, 1:nx, 1:ny, Array(A)[:, :, Int(ceil(nz/2))]; colormap=:turbo)
    # Colorbar(fig[1,1][1,2], plt)
    # # display(fig)
    # save("out3.png", fig)

    finalize_distributed(; finalize_MPI=true)
    return
end

backend = ROCBackend()
T::DataType = Float64
dims = (0, 0, 1)

main(backend, T, dims)
# main()