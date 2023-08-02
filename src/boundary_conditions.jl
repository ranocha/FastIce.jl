module BoundaryConditions

using KernelAbstractions
using Adapt

@kernel function discrete_bcs_x!(f, west_ix, east_ix, west_bc, east_bc, west_args, east_args)
    iy, iz = @index(Global, NTuple)
    @inbounds west_bc(f, iy + 1, iz + 1, west_args...)
    @inbounds east_bc(f, iy + 1, iz + 1, east_args...)
end

@kernel function discrete_bcs_y!(f, south_iy, north_iy, south_bc, north_bc, south_args, north_args)
    ix, iz = @index(Global, NTuple)
    @inbounds f[ix+1, south_iy, iz+1] = south_bc(f, ix + 1, iz + 1, south_args...)
    @inbounds f[ix+1, north_iy, iz+1] = north_bc(f, ix + 1, iz + 1, north_args...)
end

@kernel function discrete_bcs_z!(A, bot_iz, top_iz, bot_bc, top_bc, bot_args, top_args)
    ix, iy = @index(Global, NTuple)
    @inbounds A[ix+1, iy+1, bot_iz] = bot_bc(f, ix + 1, iy + 1, bot_args...)
    @inbounds A[ix+1, iy+1, top_iz] = top_bc(f, ix + 1, iy + 1, top_args...)
end

struct Shift end
struct NoShift end

struct DirichletBC{T,HasShift}
    val::T
end

DirichletBC{HasShift}(val::T) where {T,HasShift} = DirichletBC{T, HasShift}(val)

(bc::DirichletBC{<:Number})(grid, i, j) = bc.val
(bc::DirichletBC{<:AbstractArray})(grid, i, j) = @inbounds bc.val[i, j]

Adapt.adapt_structure(to, f::DirichletBC) = Adapt.adapt(to, f.val)

struct NeumannBC{T}
    val::T
end

(bc::NeumannBC{<:Number})(grid, i, j) = bc.val
(bc::NeumannBC{<:AbstractArray})(grid, i, j) = @inbounds bc.val[i, j]

Adapt.adapt_structure(to, f::NeumannBC) = Adapt.adapt(to, f.val)

@inline apply_west_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, iy, iz))
@inline apply_east_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, iy, iz))

@inline apply_south_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iz))
@inline apply_north_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iz))

@inline apply_bot_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iy))
@inline apply_top_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,NoShift}) where {T} = bc1!(f, CartesianIndex(ix, iy, iz), bc(grid, ix, iy))

@inline function apply_west_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix + 1, iy, iz)
    bc2!(f, I1, I2, 2.0 * bc(grid, iy, iz), -1.0)
    return
end

@inline function apply_east_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix - 1, iy, iz)
    bc2!(f, I1, I2, 2.0 * bc(grid, iy, iz), -1.0)
    return
end

@inline function apply_south_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy + 1, iz)
    bc2!(f, I1, I2, 2.0 * bc(grid, ix, iz), -1.0)
    return
end

@inline function apply_north_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy - 1, iz)
    bc2!(f, I1, I2, 2.0 * bc(grid, ix, iz), -1.0)
    return
end

@inline function apply_bot_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz + 1)
    bc2!(f, I1, I2, 2.0 * bc(grid, ix, iy), -1.0)
    return
end

@inline function apply_top_bc!(f, grid, ix, iy, iz, bc::DirichletBC{T,Shift}) where {T}
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz - 1)
    bc2!(f, I1, I2, 2.0 * bc(grid, ix, iy), -1.0)
    return
end


@inline function apply_west_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix + 1, iy, iz)
    bc2!(f, I1, I2, -Δ(grid, 1) * bc(grid, iy, iz), 1.0)
    return
end

@inline function apply_east_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix - 1, iy, iz)
    bc2!(f, I1, I2, Δ(grid, 1) * bc(grid, iy, iz), 1.0)
    return
end

@inline function apply_south_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy + 1, iz)
    bc2!(f, I1, I2, -Δ(grid, 2) * bc(grid, ix, iz), 1.0)
    return
end

@inline function apply_north_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy - 1, iz)
    bc2!(f, I1, I2, Δ(grid, 2) * bc(grid, ix, iz), 1.0)
    return
end

@inline function apply_bot_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz + 1)
    bc2!(f, I1, I2, -Δ(grid, 3) * bc(grid, ix, iy), 1.0)
    return
end

@inline function apply_top_bc!(f, grid, ix, iy, iz, bc::NeumannBC)
    I1, I2 = CartesianIndex(ix, iy, iz), CartesianIndex(ix, iy, iz - 1)
    bc2!(f, I1, I2, Δ(grid, 3) * bc(grid, ix, iy), 1.0)
    return
end

@inline bc1!(f, I, v) = @inbounds f[I] = v
@inline bc2!(f, I1, I2, a, b) = @inbounds f[I1] = a + b * f[I2]

end