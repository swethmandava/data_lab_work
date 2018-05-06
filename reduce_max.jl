Pkg.add("CUDAdrv")
using CUDAdrv
Pkg.add("CUDAnative")
using CUDAnative

function reduce_grid(input::CuDeviceVector{T}, output::CuDeviceVector{T}, len::Integer) where {T}
	val = 
	i = (blockIdx().x - UInt32(1)) * blockDim().x + threadIdx().x
	step = blockDim().x * gridDim().x
	while i <= len
		@inbounds val = max(val, input[i])
		i += step
	end

	val = reduce_block(val)
	if threadIdx().x == Uint32(1)
		@inbounds output[blockIdx().x] = val
	end
	return
end


# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(val::T)::T where {T}
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    # TODO: use fldmod1 (JuliaGPU/CUDAnative.jl#28)
    wid  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
    lane = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)

    # each warp performs partial reduction
    val = reduce_warp(val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(val)
    end

    return val
end



function reduce_warp(val)
	offset = CUDAnative.warpsize() / 2
	while offset > 0
		val = max(val, shfl_down_sync(val, offset))
		offset = offset/2
	end
	return val
end


"""
Reduce a large array.
"""
function gpu_reduce(input::CuVector{T}) where {T}
    len = length(input)
    output = similar(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((len + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        return maximum(Array(input))
    end

    @cuda (blocks,threads) reduce_grid(input, output, Int32(len))
    @cuda (1,1024) reduce_grid(output, output, Int32(blocks))

    return Array(output)[1]
end



# FURTHER IMPROVEMENTS:
# - use atomic memory operations
# - dynamic block/grid size based on device capabilities
# - vectorized memory access
#   devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/


# len = 512
# a = rand(Int, len)
# d_a = CuVector(a)
# d_b = similar(d_a)

# gpu_reduce(d_a, d_b)

# b = Array(g_b)
# show(a)
# show(b)

