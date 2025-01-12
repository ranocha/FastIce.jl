const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 7
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : true
const do_visu = haskey(ENV, "DO_VISU") ? parse(Bool, ENV["DO_VISU"]) : false
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(gpu_id)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Printf, Statistics, LinearAlgebra, MAT, Random, UnPack, Plots

include(joinpath(@__DIR__, "helpers2D_v3.jl"))

import ParallelStencil: INDICES
ix,iy = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro fm(A)   esc(:( $A[$ix,$iy] == fluid )) end
macro fmxy(A) esc(:( !($A[$ix,$iy] == air || $A[$ix+1,$iy] == air || $A[$ix,$iy+1] == air || $A[$ix+1,$iy+1] == air) )) end

@parallel function compute_P_τ!(∇V::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, ϕ, r::Data.Number, μ_veτ::Data.Number, Gdτ::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @fm(ϕ)*(@all(Pt) - r*Gdτ*@all(∇V))    
    @all(τxx) = @fm(ϕ)*2.0*μ_veτ*(@d_xa(Vx)/dx + @all(τxx)/Gdτ/2.0)
    @all(τyy) = @fm(ϕ)*2.0*μ_veτ*(@d_ya(Vy)/dy + @all(τyy)/Gdτ/2.0)
    @all(τxy) = @fmxy(ϕ)*2.0*μ_veτ*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/Gdτ/2.0)
    return
end

macro sm_xi(A) esc(:( !(($A[$ix,$iyi] == solid) || ($A[$ix+1,$iyi] == solid)) )) end
macro sm_yi(A) esc(:( !(($A[$ixi,$iy] == solid) || ($A[$ixi,$iy+1] == solid)) )) end
macro fm_xi(A) esc(:( 0.5*((($A[$ix,$iyi] != air)) + (($A[$ix+1,$iyi] != air))) )) end
macro fm_yi(A) esc(:( 0.5*((($A[$ixi,$iy] != air)) + (($A[$ixi,$iy+1] != air))) )) end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, ϕ, ρgx::Data.Number, ρgy::Data.Number, dτ_ρ::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn(Vx) = @sm_xi(ϕ)*( @inn(Vx) + dτ_ρ*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx) )
    @inn(Vy) = @sm_yi(ϕ)*( @inn(Vy) + dτ_ρ*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy) )
    return
end

@parallel function compute_Res!(Rx::Data.Array, Ry::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, ϕ, ρgx::Data.Number, ρgy::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx) = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry) = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    return
end

@parallel function preprocess_visu!(Vn, τII, Vx, Vy, τxx, τyy, τxy)
    @all(Vn)  = (@av_xa(Vx)*@av_xa(Vx) + @av_ya(Vy)*@av_ya(Vy))^0.5
    @all(τII) = (0.5*(@inn(τxx)*@inn(τxx) + @inn(τyy)*@inn(τyy)) + @av(τxy)*@av(τxy))^0.5
    return
end

@views function Stokes2D(inputs::InputParams2D)
    @unpack ϕ, x2rot, y2rot, x2, y2, xc, yc, lx, ly, nx, ny, α, sc = inputs
    println("Ice flow solver: lx=$(round(lx,sigdigits=4)), ly=$(round(ly,sigdigits=4)), sc=$(round(sc,sigdigits=4)), α=$(round(α,sigdigits=4))")
    # physics
    ## dimensionally independent
    # ly        = 1.0               # domain height    [m]
    μs0       = 1.0               # matrix viscosity [Pa*s]
    ρg0       = 1.0               # gravity          [Pa/m]
    ## scales
    psc       = ρg0*ly
    tsc       = μs0/psc
    vsc       = ly/tsc
    ## dimensionally dependent
    # lx        = lx_ly*ly
    ρgx       = ρg0*sin(α)
    ρgy       = ρg0*cos(α)
    # numerics
    maxiter   = 50nx         # maximum number of pseudo-transient iterations
    nchk      = nx           # error checking frequency
    nviz      = nx           # visualisation frequency
    ε_V       = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerance for divergence
    CFL       = 0.95/sqrt(2) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = ly
    Vpdτ      = min(dx,dy)*CFL
    dτ_ρ      = Vpdτ*max_lxy/Re/μs0
    Gdτ       = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ     = 1.0/(1.0/Gdτ + 1.0/μs0)
    # allocation
    Pt        = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    τxx       = @zeros(nx  ,ny  )
    τyy       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    Vn        = @zeros(nx  ,ny  )
    τII       = @zeros(nx-2,ny-2)
    # visu
    Vn_v      = @zeros(nx,ny) # visu
    τII_v     = copy(τII) # visu
    Pt_v      = copy(Pt)  # visu
    if do_save
        !ispath("../out_visu") && mkdir("../out_visu")
        matwrite("../out_visu/out_pa.mat", Dict("Phase"=> Array(ϕ), "x2rot"=> Array(x2rot), "y2rot"=> Array(y2rot), "xc"=> Array(xc), "yc"=> Array(yc), "lx"=> lx, "ly"=> ly, "sc"=> sc, "al"=> α ); compress = true)
    end
    fntsz = 7; xci, yci = xc[2:end-1], yc[2:end-1]
    opts  = (aspect_ratio=4, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), yaxis=font(fntsz,"Courier"), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
    opts2 = (linewidth=2, markershape=:circle, markersize=3,yaxis = (:log10, font(fntsz,"Courier")), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
    # iteration loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        @parallel compute_P_τ!(∇V, Pt, τxx, τyy, τxy, Vx, Vy, ϕ, r, μ_veτ,Gdτ, dx, dy)
        @parallel compute_V!(Vx, Vy, Pt, τxx, τyy, τxy, ϕ, ρgx, ρgy, dτ_ρ, dx, dy)
        iter += 1
        if iter % nchk == 0
            @parallel compute_Res!(Rx, Ry, Pt, τxx, τyy, τxy, ϕ, ρgx, ρgy, dx, dy)
            norm_Rx = norm(Rx)/psc*lx/sqrt(length(Rx))
            norm_Ry = norm(Ry)/psc*lx/sqrt(length(Ry))
            norm_∇V = norm((ϕ.==fluid).*∇V)/vsc*lx/sqrt(length(∇V))
            err_V   = maximum([norm_Rx, norm_Ry])
            err_∇V  = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/nx)
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V)
        end
        if do_visu && iter % nviz == 0
            @parallel preprocess_visu!(Vn, τII, Vx, Vy, τxx, τyy, τxy)
            Vn_v  .= Vn;  Vn_v[Vn.==0]   .= NaN
            τII_v .= τII; τII_v[τII.==0] .= NaN
            Pt_v  .= Pt;  Pt_v[Pt.==0]   .= NaN
            p1 = heatmap(xc,yc,Array(Vn_v)';    c=:batlow, title="||V||", opts...)
            p2 = heatmap(xci,yci,Array(τII_v)'; c=:batlow, title="τII", opts...)
            p3 = heatmap(xc,yc,Array(Pt_v)';    c=:viridis, title="Pressure", #=clims=(0.0, 0.6),=# opts...)
            p4 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            display(plot(p1, p2, p3, p4, size=(1e3,600), dpi=200))
        end
    end
    if do_save
        @parallel preprocess_visu!(Vn, τII, Vx, Vy, τxx, τyy, τxy)
        Vn_v  .= Vn;  Vn_v[Vn.==0]   .= NaN
        τII_v .= τII; τII_v[τII.==0] .= NaN
        Pt_v  .= Pt;  Pt_v[Pt.==0]   .= NaN
        matwrite("../out_visu/out_res.mat", Dict("Vn"=> Array(Vn), "tII"=> Array(τII), "Pt"=> Array(Pt), "xc"=> Array(xc), "yc"=> Array(yc)); compress = true)
    end
    return
end

# ---------------------

# preprocessing
inputs = preprocess("../data/arolla2D/arolla51.txt"; resx=256, do_rotate=true, fact_ny=5)

@time Stokes2D(inputs)
