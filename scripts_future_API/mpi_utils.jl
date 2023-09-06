@inline subrange(nr,bw,I,::Val{1}) = 1:bw[I]
@inline subrange(nr,bw,I,::Val{2}) = (size(nr,I)-bw[I]+1):size(nr,I)
@inline subrange(nr,bw,I,::Val{3}) = (bw[I]+1):(size(nr,I)-bw[I])

@inline split_ndrange(ndrange,ndwidth) = split_ndrange(CartesianIndices(ndrange),ndwidth)

function split_ndrange(ndrange::CartesianIndices{N},ndwidth::NTuple{N,<:Integer}) where N
    @assert all(size(ndrange) .> ndwidth.*2)
    @inline ndsubrange(I,::Val{J}) where J = ntuple(Val(N)) do idim
        if idim < I
            1:size(ndrange,idim)
        elseif idim == I
            subrange(ndrange,ndwidth,idim,Val(J))
        else
            subrange(ndrange,ndwidth,idim,Val(3))
        end
    end
    ndinner = ntuple(idim -> subrange(ndrange,ndwidth,idim,Val(3)), Val(N))
    return ntuple(Val(2N+1)) do i
        if i == 2N+1
            ndrange[ndinner...]
        else
            idim,idir = divrem(i-1,2) .+ 1
            ndrange[ndsubrange(idim,Val(idir))...]
        end
    end
end

function gather!(dst, src, comm; root=0)
    dims, _, _ = MPI.Cart_get(comm)
    dims = Tuple(dims)
    if MPI.Comm_rank(comm) == root
        # make subtype for gather
        subtype = MPI.Types.create_subarray(size(dst), size(src), (0, 0), MPI.Datatype(eltype(dst)))
        subtype = MPI.Types.create_resized(subtype, 0, size(src, 1) * Base.elsize(dst))
        MPI.Types.commit!(subtype)
        # make VBuffer for collective communication
        counts = fill(Cint(1), dims)
        displs = similar(counts)
        d = zero(Cint)
        for j in 1:dims[2]
            for i in 1:dims[1]
                displs[i, j] = d
                d += 1
            end
            d += (size(src, 2) - 1) * dims[2]
        end
        # transpose displs as cartesian communicator is row-major
        recvbuf = MPI.VBuffer(dst, vec(counts), vec(displs'), subtype)
        MPI.Gatherv!(src, recvbuf, comm; root)
    else
        MPI.Gatherv!(src, nothing, comm; root)
    end
    return
end