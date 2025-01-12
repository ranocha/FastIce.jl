using FastIce
using TinyKernels
using HDF5
using LightXML

include("load_dem.jl")
include("signed_distances.jl")
include("level_sets.jl")
include("volume_fractions.jl")
include("bcs.jl")
include("stokes.jl")
include("data_io.jl")
include("hide_communication.jl")

@views av1(A) = 0.5.*(A[1:end-1].+A[2:end])
@views inn_x(A) = A[2:end-1,:]
@views inn_y(A) = A[:,2:end-1]
@views inn(A)   = A[2:end-1,2:end-1]

const DAT = Float32

@views function main(grid_dims)

    # load DEM
    (me==0) && @info "loading DEM data from the file '$greenland_path'"
    (;x,y,bed,surface) = load_dem(greenland_path,global_region)
    (me==0) && @info "DEM resolution: $(size(bed,1)) × $(size(bed,2))"

    # compute origin and size of the domain (required for scaling and computing the grid size)
    ox,oy,oz = x[1], y[1], minimum(bed)
    lx = x[end] - ox
    ly = y[end] - oy
    lz = maximum(surface) - oz

    # shift and scale the domain before computation (center of the domain is (0,0) in x-y plane)
    δx, δy = ox + 0.5lx,oy + 0.5ly # required to avoid conversion to Vector  
    x = @. (x - δx)/lz
    y = @. (y - δy)/lz
    @. bed     = (bed     - oz)/lz
    @. surface = (surface - oz)/lz

    # run simulation
    dem_data = (;x,y,bed,surface)
    @info "running the simulation"
    run_simulation(dem_data,grid_dims,me,dims,coords,comm_cart)

    return
end

@views function run_simulation(dem_data,grid_dims,me,dims,coords,comm_cart)
    # physics
    # global domain origin and size
    ox_g, oy_g, oz_g = dem_data.x[1], dem_data.y[1], 0.0
    lx_g = dem_data.x[end] - ox_g
    ly_g = dem_data.y[end] - oy_g
    lz_g = 1.0

    ρg  = (x=0.0,y=0.0,z=1.0)

    # local domain size and origin
    lx_l,ly_l,lz_l = (lx_g,ly_g,lz_g)./dims
    ox_l,oy_l,oz_l = (ox_g,oy_g,oz_g) .+ coords.*(lx_l,ly_l,lz_l)

    # numerics
    nx,ny,nz = grid_dims
    bwidth   = (8,4,4)
    
    # preprocessing
    dx,dy,dz = lx_g/nx_g(), ly_g/ny_g(), lz_g/nz_g()
    (me==0) && @info "grid spacing: dx = $dx, dy = $dy, dz = $dz"

    xv_l = LinRange(ox_l,ox_l+lx_l,nx+1)
    yv_l = LinRange(oy_l,oy_l+ly_l,ny+1)
    zv_l = LinRange(oz_l,oz_l+lz_l,nz+1)
    xc_l,yc_l,zc_l = av1.((xv_l,yv_l,zv_l))
    
    # PT params
    r          = 0.7
    lτ_re_mech = 0.5min(lx_g,ly_g,lz_g)/π
    vdτ        = min(dx,dy,dz)/sqrt(10.1)
    θ_dτ       = lτ_re_mech*(r+4/3)/vdτ
    nudτ       = vdτ*lτ_re_mech
    dτ_r       = 1.0/(θ_dτ+1.0)

    # fields allocation
    # level set
    Ψ = (
        not_solid = field_array(DAT,nx+1,ny+1), # fluid
        not_air   = field_array(DAT,nx+1,ny+1), # liquid
    )
    wt = (
        not_solid = (
            c  = field_array(DAT,nx  ,ny  ),
            x  = field_array(DAT,nx+1,ny  ),
            y  = field_array(DAT,nx  ,ny+1),
            xy = field_array(DAT,nx-1,ny-1),
        ),
        not_air = (
            c  = field_array(DAT,nx  ,ny  ),
            x  = field_array(DAT,nx+1,ny  ),
            y  = field_array(DAT,nx  ,ny+1),
            xy = field_array(DAT,nx-1,ny-1),
        )
    )
    # mechanics
    Pr = field_array(DAT,nx,ny)
    τ  = (
        xx = field_array(DAT,nx  ,ny  ),
        yy = field_array(DAT,nx  ,ny  ),
        xy = field_array(DAT,nx-1,ny-1),
    )
    V = (
        x = field_array(DAT,nx+1,ny),
        y = field_array(DAT,nx,ny+1),
    )
    ηs = field_array(DAT,nx,ny)
    # residuals
    Res = (
        Pr = field_array(DAT,nx,ny),
        V = (
            x = field_array(DAT,nx-1,ny-2),
            y = field_array(DAT,nx-2,ny-1),
        )
    )
    # visualisation
    Vmag = field_array(DAT,nx-2,ny-2)
    τII  = field_array(DAT,nx-2,ny-2)
    Ψav  = (
        not_solid = field_array(DAT,nx-2,ny-2),
        not_air   = field_array(DAT,nx-2,ny-2),
    )

    # initialisation
    for comp in eachindex(V) fill!(V[comp],0.0) end
    for comp in eachindex(τ) fill!(τ[comp],0.0) end
    fill!(Pr,0.0)
    fill!(ηs,1.0)

    # compute level sets from DEM data
    dem_grid = (dem_data.x,dem_data.y)
    Ψ_grid   = (xv_l,yv_l,zv_l)
    
    (me==0) && @info "computing the level set for the ice surface"
    compute_level_set_from_dem!(Ψ.not_air,to_device(dem_data.surface),dem_grid,Ψ_grid)

    (me==0) && @info "computing the level set for the bedrock surface"
    compute_level_set_from_dem!(Ψ.not_solid,to_device(dem_data.bed),dem_grid,Ψ_grid)
    TinyKernels.device_synchronize(get_device())
    @. Ψ.not_solid*= -1.0
    TinyKernels.device_synchronize(get_device())

    (me==0) && @info "computing volume fractions from level sets"
    for phase in eachindex(Ψ)
        compute_volume_fractions_from_level_set!(wt[phase],Ψ[phase],dx,dy,dz)
    end
    
    (me==0) && @info "iteration loop"
    for iter in 1:1000
        (me==0) && println("  iter: $iter")
        update_σ!(Pr,τ,V,ηs,wt,r,θ_dτ,dτ_r,dx,dy,dz)
        update_V!(V,Pr,τ,ηs,wt,nudτ,ρg,dx,dy,dz;bwidth)
    end

    (me==0) && @info "saving results on disk"
    dim_g = (nx_g()-2, ny_g()-2, nz_g()-2)
    update_vis_fields!(Vmag,τII,Ψav,V,τ,Ψ)
    out_h5 = "results.h5"
    ndrange = CartesianIndices(( (coords[1]*(nx-2) + 1):(coords[1]+1)*(nx-2),
                                 (coords[2]*(ny-2) + 1):(coords[2]+1)*(ny-2),
                                 (coords[3]*(nz-2) + 1):(coords[3]+1)*(nz-2) ))
    fields = Dict("LS_ice"=>Ψav.not_air,"LS_bed"=>Ψav.not_solid"Vmag"=>Vmag,"TII"=>τII,"Pr"=>inn(Pr))
    (me==0) && @info "saving HDF5 file"
    write_h5(out_h5,fields,dim_g,ndrange,comm_cart,MPI.Info())

    if me==0
        @info "saving XDMF file..."
        write_xdmf("results.xdmf3",out_h5,fields,(xc_l[2],yc_l[2],zc_l[2]),(dx,dy,dz),dim_g)
    end

    return
end

@tiny function _kernel_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    ix,iy,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    @inbounds if isin(Ψ.not_air)
        pav = 0.0
        for idz = 1:2, idy=1:2, idx = 1:2
            pav += Ψ.not_air[ix+idx,iy+idy,iz+idz]
        end
        Ψav.not_air[ix,iy,iz] = pav/8
    end
    @inbounds if isin(Ψ.not_solid
        pav = 0.0
        for idz = 1:2, idy=1:2, idx = 1:2
            pav += Ψ.not_solidix+idx,iy+idy,iz+idz]
        end
        Ψav.not_solidix,iy,iz] = pav/8
    end
    @inbounds if isin(Vmag)
        vxc = 0.5*(V.x[ix+1,iy+1,iz+1] + V.x[ix+2,iy+1,iz+1])
        vyc = 0.5*(V.y[ix+1,iy+1,iz+1] + V.y[ix+1,iy+2,iz+1])
        vzc = 0.5*(V.z[ix+1,iy+1,iz+1] + V.z[ix+1,iy+1,iz+2])
        Vmag[ix,iy,iz] = sqrt(vxc^2 + vyc^2 + vzc^2)
    end
    @inbounds if isin(τII)
        τxyc = 0.25*(τ.xy[ix,iy,iz]+τ.xy[ix+1,iy,iz]+τ.xy[ix,iy+1,iz]+τ.xy[ix+1,iy+1,iz])
        τxzc = 0.25*(τ.xz[ix,iy,iz]+τ.xz[ix+1,iy,iz]+τ.xz[ix,iy,iz+1]+τ.xz[ix+1,iy,iz+1])
        τyzc = 0.25*(τ.yz[ix,iy,iz]+τ.yz[ix,iy+1,iz]+τ.yz[ix,iy,iz+1]+τ.yz[ix,iy+1,iz+1])
        τII[ix,iy,iz] = sqrt(0.5*(τ.xx[ix+1,iy+1,iz+1]^2 + τ.yy[ix+1,iy+1,iz+1]^2 + τ.zz[ix+1,iy+1,iz+1]^2) + τxyc^2 + τxzc^2 + τyzc^2)
    end
    return
end

const _update_vis_fields! = Kernel(_kernel_update_vis_fields!,get_device())

function update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ)
    wait(_update_vis_fields!(Vmag, τII, Ψav, V, τ, Ψ; ndrange=axes(Vmag)))
    return
end

main((1024,1024,64))