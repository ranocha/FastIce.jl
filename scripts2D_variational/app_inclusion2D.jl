using FastIce
using TinyKernels
using CairoMakie
using ElasticArrays
using Printf

include("bcs.jl")
include("helpers_tmp.jl")
include("level_sets.jl")
include("stokes.jl")
include("volume_fractions.jl")

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views inn_x(A) = A[2:end-1,:]
@views inn_y(A) = A[:,2:end-1]
@views inn(A) = A[2:end-1,2:end-1]

@views function runsim(::Type{DAT}; nx=127) where {DAT}
    # physics
    # lx, ly   = 2.0, 1.0
    lx, ly   = 1.0, 1.0
    ox, oy   = -0.5lx, -0.5ly
    # xb1, yb1 = ox + 0.5lx, oy + 0.0ly
    # xb2, yb2 = ox + 0.5lx, oy + 3.0ly
    xb1, yb1 = ox + 0.5lx, oy + 0.5ly
    rinc     = 0.14ly
    # rair     = 2.3ly
    ηs0      = 1.0
    ebg      = 1.0
    ρg0      = 0.0
    α        = 0.0
    npow     = 1
    # numerics
    ny       = ceil(Int, (nx + 1) * ly / lx) - 1
    maxiter  = 40nx
    ncheck   = 2nx
    ϵtol     = (1e-6, 1e-6, 1e-6)
    nt       = 1
    χ        = 1.0
    # preprocessing
    dx, dy   = lx / nx, ly / ny
    xv, yv   = LinRange(ox, ox + lx, nx + 1), LinRange(oy, oy + ly, ny + 1)
    xc, yc   = av1(xv), av1(yv)
    mc1      = to_device(make_marker_chain_circle(Point(xb1, yb1), rinc, min(dx, dy)))
    # mc2      = to_device(make_marker_chain_circle(Point(xb2, yb2), rair, min(dx, dy)))
    ρg       = (x=ρg0 .* sin(α), y=ρg0 .* cos(α))
    mpow     = -(1 - 1 / npow)
    # PT parameters
    r        = 0.7
    re_mech  = 6π
    lτ       = min(lx, ly)
    vdτ      = min(dx, dy) / sqrt(2.1)
    θ_dτ     = lτ * (r + 4 / 3) / (re_mech * vdτ)
    nudτ     = vdτ * lτ / re_mech
    dτ_r     = 1.0 / (θ_dτ + 1.0)
    # level set
    Ψ  = (
        not_air = field_array(DAT, nx + 1, ny + 1),  # liquid
    )
    wt = (
        not_solid = volfrac_field(DAT, nx, ny), # fluid
        not_air   = volfrac_field(DAT, nx, ny), # liquid
    )
    # mechanics
    Pr = scalar_field(DAT, nx, ny)
    τ  = tensor_field(DAT, nx, ny)
    ε  = tensor_field(DAT, nx, ny)
    V  = vector_field(DAT, nx, ny)
    ηs = scalar_field(DAT, nx, ny)
    τII = scalar_field(DAT, nx, ny)
    εII = scalar_field(DAT, nx, ny)
    # residuals
    Res = (
        Pr = scalar_field(DAT, nx    , ny    ),
        V  = vector_field(DAT, nx - 2, ny - 2),
    )
    # visualisation
    Vmag = field_array(DAT, nx - 2, ny - 2)
    Ψav = (
        not_solid = field_array(DAT, nx - 2, ny - 2),
        not_air   = field_array(DAT, nx - 2, ny - 2),
    )
    # initial and boundary conditions
    @info "computing the level set for the inclusion"
    for comp in eachindex(Ψ) fill!(Ψ[comp], 1.0) end
    init!(Pr, τ, V, ηs, ebg, ηs0, xv, yv)
    fill!(τII, 0.0)
    fill!(εII, 0.0)
    Ψ.not_air .= Inf # needs init now
    compute_levelset!(Ψ.not_air, xv, yv, mc1)
    # compute_levelset!(Ψ.not_air, xv, yv, mc2)
    Ψ.not_air .= .-Ψ.not_air

    @info "computing volume fractions from level sets"
    compute_volume_fractions_from_level_set!(wt.not_air, Ψ.not_air, dx, dy)
    for comp in eachindex(wt.not_solid) fill!(wt.not_solid[comp], 1.0) end

    update_vis!(Vmag, Ψav, V, Ψ)
    # convergence history
    iter_evo = Float64[]
    errs_evo = ElasticArray{Float64}(undef, length(ϵtol), 0)
    # figures
    fig = Figure(resolution=(2500, 1600), fontsize=32)
    ax = (
        Pr  =Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="p"),
        τII =Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="τII"),
        Vmag=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|v|"),
        εII =Axis(fig[2, 2][1, 1]; aspect=DataAspect(), title="εII"),
        wt  =Axis(fig[1, 3][1, 1]; aspect=DataAspect(), title="Volume fraction"),
        errs=Axis(fig[2, 3]      ; yscale=log10, title="Convergence", xlabel="#iter/ny", ylabel="error"),
    )
    plt = (
        fields=(
            Pr  =heatmap!(ax.Pr  , xc, yc, to_host(Pr  ); colormap=:turbo),
            τII =heatmap!(ax.τII , xc, yc, to_host(τII ); colormap=:turbo),
            Vmag=heatmap!(ax.Vmag, xc, yc, to_host(Vmag); colormap=:turbo),
            εII =heatmap!(ax.εII , xc, yc, to_host(εII ); colormap=:turbo),
            wt  =heatmap!(ax.wt  , xc, yc, to_host(wt.not_air.c); colormap=Reverse(:grays)),
        ),
        errs=[scatterlines!(ax.errs, Point2.(iter_evo, errs_evo[ir, :])) for ir in eachindex(ϵtol)],
    )
    Colorbar(fig[1, 1][1, 2], plt.fields.Pr  )
    Colorbar(fig[1, 2][1, 2], plt.fields.τII )
    Colorbar(fig[2, 1][1, 2], plt.fields.Vmag)
    Colorbar(fig[2, 2][1, 2], plt.fields.εII )
    Colorbar(fig[1, 3][1, 2], plt.fields.wt  )
    display(fig)

    @info "running simulation 🚀"
    for it in 1:nt
        @printf "it # %d\n" it
        # iteration loop
        empty!(iter_evo); resize!(errs_evo, length(ϵtol), 0)
        iter = 0; errs = 2.0 .* ϵtol
        while any(errs .>= ϵtol) && (iter += 1) <= maxiter
            update_σ!(Pr, ε, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
            compute_invariants!(εII, τII, ε, τ, ηs, χ, mpow)
            update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy)
            if iter % ncheck == 0
                compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy)
                errs = (maximum(abs.(Res.V.x)), maximum(abs.(Res.V.y)), maximum(abs.(Res.Pr)))
                @printf "  iter/nx # %2.1f, errs: [ Vx = %1.3e, Vy = %1.3e, Pr = %1.3e ]\n" iter / nx errs...
                push!(iter_evo, iter / nx); append!(errs_evo, errs)
                # visu
                for ir in eachindex(plt.errs)
                    plt.errs[ir][1] = Point2.(iter_evo, errs_evo[ir, :])
                end
                autolimits!(ax.errs)
                update_vis!(Vmag, Ψav, V, Ψ)
                plt.fields[1][3] = to_host(to_host(Pr))
                plt.fields[2][3] = to_host(to_host(τII))
                plt.fields[3][3] = to_host(to_host(Vmag))
                # plt.fields[4][3] = to_host(to_host(εII))
                plt.fields[4][3] = to_host(to_host(log10.(ηs)))
                plt.fields[5][3] = to_host(to_host(wt.not_air.c))
                # plt.fields[4][3] = to_host(to_host(Ψ.not_air))
                display(fig)
            end
        end
    end
    return
end

runsim(Float64, nx=127)