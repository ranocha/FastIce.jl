const USE_GPU = haskey(ENV,"USE_GPU") ? parse(Bool,ENV["USE_GPU"]) : false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA,Float64,2)
else
    @init_parallel_stencil(Threads,Float64,2)
end
using ElasticArrays,Printf
using Plots,Plots.Measures
default(size=(800,500),framestyle=:box,label=false,grid=false,margin=3mm,lw=6,labelfontsize=11,tickfontsize=11,titlefontsize=11)
# using GLMakie
# include("vis_helpers.jl")

@views amean1(A) = 0.5.*(A[1:end-1] .+ A[2:end])
@views ameany(A) = 0.5.*(A[:,1:end-1] .+ A[:,2:end])
@views ameanx(A) = 0.5.*(A[1:end-1,:] .+ A[2:end,:])

macro my_maxloc(A) esc(:( max.($A[$ixi-1,$iyi-1],$A[$ixi-1,$iyi],$A[$ixi-1,$iyi+1],
                               $A[$ixi  ,$iyi-1],$A[$ixi  ,$iyi],$A[$ixi  ,$iyi+1],
                               $A[$ixi+1,$iyi-1],$A[$ixi+1,$iyi],$A[$ixi+1,$iyi+1]) )) end

import ParallelStencil: INDICES
ix,iy = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)
macro d_xii(A) esc(:( $A[$ixi+1,$iyi  ]-$A[$ixi,$iyi] )) end
macro d_yii(A) esc(:( $A[$ixi  ,$iyi+1]-$A[$ixi,$iyi] )) end

@parallel function update_iter_params!(ητ,η)
    @inn(ητ) = @my_maxloc(η)
    return
end

@parallel_indices (ix) function bc_y!(A)
    A[ix,1]   = A[ix,2]
    A[ix,end] = A[ix,end-1]
    return
end

@parallel_indices (iy) function bc_x!(A)
    A[1  ,iy] = A[2    ,iy]
    A[end,iy] = A[end-1,iy]
    return
end

@parallel function update_normal_τ!(Pr,dPr,εxx,εyy,εxyv,Vx,Vy,∇V,η,r,θ_dτ,dx,dy)
    @all(∇V)   = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(dPr)  = -@all(∇V)
    @all(Pr)   = @all(Pr) + @all(dPr)*@all(η)*r/θ_dτ
    @all(εxx)  = @d_xa(Vx)/dx - @all(∇V)/3.0
    @all(εyy)  = @d_ya(Vy)/dy - @all(∇V)/3.0
    @inn(εxyv) = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    return
end

@parallel function update_shear_τ!(τxx,τyy,τxy,τII,εxx,εyy,εxy,εxyv,η,dτ_r)
    @all(εxy)  = @av(εxyv)
    @all(τxx)  = @all(τxx) + (-@all(τxx) + 2.0*@all(η)*@all(εxx))*dτ_r
    @all(τyy)  = @all(τyy) + (-@all(τyy) + 2.0*@all(η)*@all(εyy))*dτ_r
    @all(τxy)  = @all(τxy) + (-@all(τxy) + 2.0*@all(η)*@all(εxy))*dτ_r
    @all(τII)  = sqrt(0.5*(@all(τxx)*@all(τxx) + @all(τyy)*@all(τyy)) + @all(τxy)*@all(τxy))
    return
end

@parallel function update_τxy!(τxyv,τxy)
    @inn(τxyv) = @av(τxy)
    return
end

@parallel function update_velocities!(Vx,Vy,Pr,τxx,τyy,τxyv,ητ,ρgx,ρgy,nudτ,dx,dy)
    @inn_x(Vx) = @inn_x(Vx) + (-@d_xa(Pr)/dx + @d_xa(τxx)/dx + @d_yi(τxyv)/dy - @all(ρgx))*nudτ/@av_xa(ητ)
    @inn_y(Vy) = @inn_y(Vy) + (-@d_ya(Pr)/dy + @d_ya(τyy)/dy + @d_xi(τxyv)/dx - @all(ρgy))*nudτ/@av_ya(ητ)
    return
end

@parallel_indices (ix) function bc_Vx!(Vx)
    Vx[ix,1] = 1.0/3.0*Vx[ix,2]
    return
end

@parallel_indices (ix) function bc_Vy!(Vy,η,Pr,dy)
    Vy[ix,end] = Vy[ix,end-1] + 0.5*dy/η[ix,end]*(Pr[ix,end] + 1.0/3.0*(-Pr[ix,end-1] + 2.0*η[ix,end-1]*(Vy[ix,end-1] - Vy[ix,end-2])/dy))
    return
end

@parallel function compute_residuals!(r_Vx,r_Vy,Pr,τxx,τyy,τxyv,ρgx,ρgy,dx,dy)
    @all(r_Vx) = -@d_xi(Pr)/dx + @d_xi(τxx)/dx + @d_yii(τxyv)/dy - @inn_y(ρgx)
    @all(r_Vy) = -@d_yi(Pr)/dy + @d_yi(τyy)/dy + @d_xii(τxyv)/dx - @inn_x(ρgy)
    return
end

@parallel_indices (ix,iy) function compte_η_ρg!(η,ρgy_c,phase,ηb,xc,yc,x0,y0c,y0d,r_dep,δ_sd,η0_air,η0_ice,ρg0_air,ρg0_ice)
    if ix<=size(η,1) && iy<=size(η,2)
        sd_air = sqrt((xc[ix]-x0)^2 + 5*(yc[iy]-y0d)^2)-r_dep
        t_air  = 0.5*(tanh(-sd_air/δ_sd) + 1)
        t_ice  = 1.0 - t_air
        η[ix,iy]     = t_ice*η0_ice  + t_air*η0_air
        ρgy_c[ix,iy] = t_ice*ρg0_ice + t_air*ρg0_air
        phase[ix,iy] =  1.0 - t_air
        ηb[ix,iy]    = (1.0 - t_air)*1e12 + t_air*1.0
    end
    return
end

function main()
    # physics
    lx,ly      = 20.0,10.0
    η0         = (ice = 1.0 , air = 1e-6)
    ρg0        = (ice = 0.9 , air = 0.0 )
    r_dep      = 1.5*min(lx,ly)
    x0,y0c,y0d = 0.0,0.0*ly,1.4*ly
    # numerics
    nx         = 200
    ny         = ceil(Int,nx*ly/lx)
    ϵtol       = (1e-6,1e-6,1e-6)
    maxiter    = 100max(nx,ny)
    ncheck     = ceil(Int,5max(nx,ny))
    r          = 0.7
    re_mech    = 3π
    δ_sd       = 0.06
    # preprocessing
    dx,dy      = lx/nx,ly/ny
    xv,yv      = LinRange(-lx/2,lx/2,nx+1),LinRange(0,ly,ny+1)
    xc,yc      = amean1(xv),amean1(yv)
    lτ         = min(lx,ly)
    vdτ        = 0.9*min(dx,dy)/sqrt(2.1)
    θ_dτ       = lτ*(r+2.0)/(re_mech*vdτ)
    nudτ       = vdτ*lτ/re_mech
    dτ_r       = 1.0/(θ_dτ + 1.0)
    # array allocation
    Vx         = @zeros(nx+1,ny  )
    Vy         = @zeros(nx  ,ny+1)
    Pr         = @zeros(nx  ,ny  )
    ∇V         = @zeros(nx  ,ny  )
    τxx        = @zeros(nx  ,ny  )
    τyy        = @zeros(nx  ,ny  )
    τxy        = @zeros(nx  ,ny  )
    τxyv       = @zeros(nx+1,ny+1)
    εxx        = @zeros(nx  ,ny  )
    εyy        = @zeros(nx  ,ny  )
    εxy        = @zeros(nx  ,ny  )
    εxyv       = @zeros(nx+1,ny+1)
    η          = @zeros(nx  ,ny  )
    τII        = @zeros(nx  ,ny  )
    Vmag       = @zeros(nx  ,ny  )
    dPr        = @zeros(nx  ,ny  )
    r_Vx       = @zeros(nx-1,ny-2)
    r_Vy       = @zeros(nx-2,ny-1)
    ρgy_c      = @zeros(nx  ,ny  )
    ρgx        = @zeros(nx-1,ny  )
    ρgy        = @zeros(nx  ,ny-1)
    phase      = @zeros(nx  ,ny  )
    ηb         = @zeros(nx  ,ny  )
    ητ         = @zeros(nx  ,ny  )
    # initialisation
    @parallel compte_η_ρg!(η,ρgy_c,phase,ηb,xc,yc,x0,y0c,y0d,r_dep,δ_sd,η0.air,η0.ice,ρg0.air,ρg0.ice)
    ρgy .= ameany(ρgy_c)
    Pr  .= Data.Array(reverse(cumsum(reverse(Array(ρgy_c),dims=2),dims=2).*dy,dims=2))
    opts = (aspect_ratio=1, xlims=extrema(xc), ylims=extrema(yc), c=:turbo, framestyle=:box)
    mask = Array(phase); mask[mask.<0.7].=NaN
    iter_evo=Float64[]; errs_evo=ElasticMatrix{Float64}(undef,length(ϵtol),0)
    errs = 2.0.*ϵtol; iter = 1
    resize!(iter_evo,0); resize!(errs_evo,length(ϵtol),0)
    while any(errs .>= ϵtol) && iter <= maxiter
        @parallel update_iter_params!(ητ,η)
        @parallel (1:size(ητ,1)) bc_y!(ητ)
        @parallel (1:size(ητ,2)) bc_x!(ητ)
        @parallel update_normal_τ!(Pr,dPr,εxx,εyy,εxyv,Vx,Vy,∇V,η,r,θ_dτ,dx,dy)
        @parallel update_shear_τ!(τxx,τyy,τxy,τII,εxx,εyy,εxy,εxyv,η,dτ_r)
        @parallel update_τxy!(τxyv,τxy)
        @parallel update_velocities!(Vx,Vy,Pr,τxx,τyy,τxyv,ητ,ρgx,ρgy,nudτ,dx,dy)
        @parallel (1:size(Vx,1)) bc_Vx!(Vx)
        @parallel (1:size(Vy,1)) bc_Vy!(Vy,η,Pr,dy)
        if iter % ncheck == 0
            @parallel compute_residuals!(r_Vx,r_Vy,Pr,τxx,τyy,τxyv,ρgx,ρgy,dx,dy)
            errs = maximum.((abs.(r_Vx),abs.(r_Vy),abs.(dPr)))
            push!(iter_evo,iter/max(nx,ny));append!(errs_evo,errs)
            @printf("  iter/nx=%.3f,errs=[ %1.3e, %1.3e, %1.3e ] \n",iter/max(nx,ny),errs...)
        end
        iter += 1
    end
    # visualisation
    mask = Array(phase); mask[mask.<0.7].=NaN
    Vmag .= sqrt.(ameanx(Vx).^2 + ameany(Vy).^2)
    p1=heatmap(xc,yc,mask' .* Array(Pr)',title="Pressure";opts...)
    p3=heatmap(xc,yc,mask' .* Array(τII)',title="τII";opts...)
    p2=heatmap(xc,yc,mask' .* Array(Vmag)',title="Vmag";opts...)
    display(plot(p1,p2,p3,layout=(2,2)))
    return
end

main()
