const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false
const gpu_id  = haskey(ENV, "GPU_ID" ) ? parse(Int , ENV["GPU_ID" ]) : 0
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(gpu_id)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT, Random

import ParallelStencil: INDICES
ix,iy = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)

const air   = 0.0
const fluid = 1.0
const solid = 2.0

macro fm(A)      esc(:( $A[$ix,$iy] == fluid )) end
macro fmxy_xi(A) esc(:( !(($A[$ix,$iy] == air && $A[$ix,$iy+1] == air) || ($A[$ix+1,$iy] == air && $A[$ix+1,$iy+1] == air)) )) end
macro fmxy_yi(A) esc(:( !(($A[$ix,$iy] == air && $A[$ix+1,$iy] == air) || ($A[$ix,$iy+1] == air && $A[$ix+1,$iy+1] == air)) )) end

@parallel function compute_P_τ!(∇V::Data.Array, Pt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, ϕ, r::Data.Number, μ_veτ::Data.Number, Gdτ::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @fm(ϕ)*(@all(Pt) - r*Gdτ*@all(∇V))    
    @all(τxx) = @fm(ϕ)*2.0*μ_veτ*(@d_xa(Vx)/dx + @all(τxx)/Gdτ/2.0)
    @all(τyy) = @fm(ϕ)*2.0*μ_veτ*(@d_ya(Vy)/dy + @all(τyy)/Gdτ/2.0)
    @all(τxy) = @fmxy_xi(ϕ)*@fmxy_yi(ϕ)*2.0*μ_veτ*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx) + @all(τxy)/Gdτ/2.0)
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
    @all(Rx)  = @sm_xi(ϕ)*(@d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx - @fm_xi(ϕ)*ρgx)
    @all(Ry)  = @sm_yi(ϕ)*(@d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy - @fm_yi(ϕ)*ρgy)
    return
end

function is_in_plane(x,y,tanβ,el)
    return y < tanβ*x + el
end

@parallel_indices (ix,iy) function init_ϕ!(ϕ,gl,el,tanβ,dx,dy,lx,ly)
    xc,yc = dx*ix-dx/2-lx/2, dy*iy-dy/2
    if checkbounds(Bool,ϕ,ix,iy)
        if is_in_plane(xc,yc,tanβ,gl)
            ϕ[ix,iy] = fluid
        end
        if is_in_plane(xc,yc,tanβ,el)
            ϕ[ix,iy] = solid
        end
    end
    return
end

@parallel_indices (iy) function bc_x!(Vx,Vy,Pt,ϕ,del)
    if iy <= size(Vx,2)
        if ϕ[1,iy] == fluid
            Vx[1,iy]   = Vx[end-1,iy-del]
            Vy[1,iy]   = Vy[end-1,iy-del]
            Vy[1,iy+1] = Vy[end-1,iy+1-del]
            Pt[1,iy]   = Pt[end-1,iy-del]
        end
        if ϕ[end,iy] == fluid
            Vx[end,iy]   = Vx[2,iy+del]
            Vy[end,iy]   = Vy[2,iy+del]
            Vy[end,iy+1] = Vy[2,iy+1+del]
            Pt[end,iy]   = Pt[2,iy+del]
        end
    end
    return
end

@views function Stokes2D()
    # physics
    ## dimensionally independent
    ly        = 1.0          # domain height    [m]
    μs0       = 1.0          # matrix viscosity [Pa*s]
    ρg0       = 1.0          # gravity          [Pa/m]
    ## scales
    psc       = ρg0*ly
    tsc       = μs0/psc
    vsc       = ly/tsc
    ## nondimensional parameters
    lx_ly     = 1.0
    gl_ly     = 0.75
    el_ly     = 0.25
    amp_ly    = 1/25
    α         = -π/12
    β         = -0π/12
    tanβ      = tan(β)
    ωly       = 10π
    ## dimensionally dependent
    lx        = lx_ly*ly
    gl        = gl_ly*ly
    el        = el_ly*ly
    amp       = amp_ly*ly
    ρgx       = ρg0*sin(α)
    ρgy       = ρg0*cos(α)
    ω         = ωly/ly
    # numerics
    ny        = 255
    nx        = ceil(Int,lx_ly*ny)
    nx, ny    = nx-1, ny-1
    maxiter   = 50ny         # maximum number of pseudo-transient iterations
    nchk      = 2ny          # error checking frequency
    nviz      = 2ny          # visualisation frequency
    ε_V       = 1e-8         # nonlinear absolute tolerance for momentum
    ε_∇V      = 1e-8         # nonlinear absolute tolerance for divergence
    CFL       = 0.95/sqrt(2) # stability condition
    Re        = 2π           # Reynolds number                     (numerical parameter #1)
    r         = 1.0          # Bulk to shear elastic modulus ratio (numerical parameter #2)
    # preprocessing
    dx, dy    = lx/nx, ly/ny # cell sizes
    max_lxy   = abs(gl-el)
    Vpdτ      = min(dx,dy)*CFL
    dτ_ρ      = Vpdτ*max_lxy/Re/μs0
    Gdτ       = Vpdτ^2/dτ_ρ/(r+2.0)
    μ_veτ     = 1.0/(1.0/Gdτ + 1.0/μs0)
    Xc, Yc    = LinRange(-(lx-dx)/2,(lx-dx)/2,nx  ), LinRange(dy/2,ly-dy/2,ny  )
    Xv, Yv    = LinRange(- lx/2    , lx/2    ,nx+1), LinRange(0   ,ly     ,ny+1)
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
    ϕ         = air*@ones(nx,ny)
    Vx_v      = copy(Vx) # visu
    Vy_v      = copy(Vy) # visu
    Pt_v      = copy(Pt) # visu
    @parallel init_ϕ!(ϕ,gl,el,tanβ,dx,dy,lx,ly)
    del       = findfirst(y->y==fluid,ϕ[1,:])-findfirst(y->y==fluid,ϕ[end,:])
    # iteration loop
    err_V=2*ε_V; err_∇V=2*ε_∇V; iter=0; err_evo1=[]; err_evo2=[]
    while !((err_V <= ε_V) && (err_∇V <= ε_∇V)) && (iter <= maxiter)
        @parallel compute_P_τ!(∇V, Pt, τxx, τyy, τxy, Vx, Vy, ϕ, r, μ_veτ,Gdτ, dx, dy)
        @parallel compute_V!(Vx, Vy, Pt, τxx, τyy, τxy, ϕ, ρgx, ρgy, dτ_ρ, dx, dy)
        @parallel bc_x!(Vx,Vy,Pt,ϕ,del)
        iter += 1
        if iter % nchk == 0
            @parallel compute_Res!(Rx, Ry, Pt, τxx, τyy, τxy, ϕ, ρgx, ρgy, dx, dy)
            norm_Rx = norm(Rx[2:end-1,:])/psc*lx/sqrt(length(Rx[2:end-1,:]))
            norm_Ry = norm(Ry[2:end-1,:])/psc*lx/sqrt(length(Ry[2:end-1,:]))
            norm_∇V = norm((ϕ[2:end-1,:].==fluid).*∇V[2:end-1,:])/vsc*lx/sqrt(length(∇V[2:end-1,:]))
            err_V   = maximum([norm_Rx, norm_Ry])
            err_∇V  = norm_∇V
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter/ny)
            @printf("# iters = %d, err_V = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e], err_∇V = %1.3e \n", iter, err_V, norm_Rx, norm_Ry, err_∇V)
        end
        if iter % nviz == 0
            Vx_v .= Vx; Vx_v[Vx.==0] .= NaN
            Vy_v .= Vy; Vy_v[Vy.==0] .= NaN
            Pt_v .= Pt; Pt_v[Pt.==0] .= NaN
            fntsz = 7
            opts  = (aspect_ratio=1, xlims=(Xv[1],Xv[end]), ylims=(Yc[1],Yc[end]), yaxis=font(fntsz,"Courier"), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            opts2 = (linewidth=2, markershape=:circle, markersize=3,yaxis = (:log10, font(fntsz,"Courier")), xaxis=font(fntsz,"Courier"), framestyle=:box, titlefontsize=fntsz, titlefont="Courier")
            p1 = heatmap(Xv,Yc,Array(Vx_v)'; c=:batlow, title="Vx", opts...)
            p2 = heatmap(Xc,Yv,Array(Vy_v)'; c=:batlow, title="Vy", opts...)
            p3 = heatmap(Xc,Yc,Array(Pt_v)'; c=:viridis, title="Pressure", clims=(0.0, 0.6), opts...)
            p4 = plot(err_evo2,err_evo1; legend=false, xlabel="# iterations/nx", ylabel="log10(error)", labels="max(error)", opts2...)
            Vx_yslice = Array(Vx[round(Int,nx/2),:])
            Yc_tr     = collect(Yc.-el)
            Vx_exact  = ρgx./μs0.*Yc_tr.*(Yc_tr./2 .- (gl-el))
            p5 = plot(Yc,[Vx_yslice Vx_exact]; label=["numerical" "exact"], xlims=(0.25,0.75-dy),ylims=(0.0,0.04),linewidth=2)
            display(plot(p1, p2, p3, p4, p5, size=(1e3,600), dpi=200))
        end
    end
    return
end

Stokes2D()
