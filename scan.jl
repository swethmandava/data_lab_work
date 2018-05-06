# Work-inefficient inclusive scan
# - uses shared memory to reduce
#
# Based on http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
Pkg.add("CUDAdrv")
Pkg.add("CUDAnative")
Pkg.update()
using CUDAdrv, CUDAnative

blockSize = 1024
function cpu_exclusive_scan(data::Matrix{T}) where {T}
    cols = size(data,2)
    for col in 1:cols
        accum = zero(T)
        rows = size(data,1)
        for row in 1:size(data,1)
            temp = data[row, col]
            data[row,col] = accum
            accum = accum + temp
        end
    end

    return
end

function partial_exclusive_scan(d_list::CuDeviceMatrix{T}, d_block_sums::CuDeviceMatrix{T}, numElems::Integer, num_features::Integer) where {T}
    

    tid = threadIdx().x
    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    shmem = @cuDynamicSharedMem(T, blockDim().x * num_features)

    if (id > numElems)
        for feature_id in 1:num_features
            shmem[(tid - 1) * num_features + feature_id] = 0
        end
    else
        for feature_id in 1:num_features
            shmem[(tid - 1) * num_features + feature_id] = d_list[id, feature_id]
        end
    end

    sync_threads()

    i = 2
    neighbor_offset = 1
    while (i <= blockDim().x)

        if (tid % i == 0)
            for feature_id in 1:num_features
                shmem[(tid - 1) * num_features + feature_id] += shmem[(tid - neighbor_offset - 1) * num_features + feature_id]
            end
        end

        sync_threads()
        i = i << 1
        neighbor_offset = neighbor_offset << 1
    end



    i = neighbor_offset
    neighbor_offset = neighbor_offset >> 1

    if (tid == (blockDim().x))
        for feature_id in 1:num_features
            d_block_sums[blockIdx().x, feature_id] = shmem[(tid - 1) * num_features + feature_id]
            shmem[(tid - 1) * num_features + feature_id] = 0
        end
    end

    sync_threads()

    while (i >= 2)
        if ((tid % i) == 0)
            for feature_id in 1:num_features
                old_neighbor = shmem[(tid - neighbor_offset - 1) * num_features + feature_id]
                shmem[(tid - neighbor_offset - 1) * num_features + feature_id] = shmem[(tid - 1) * num_features + feature_id]
                shmem[(tid - 1) * num_features + feature_id] += old_neighbor
            end
        end
        i = i >> 1
        neighbor_offset = neighbor_offset >> 1
        sync_threads()
    end

    if (id <= numElems)
        for feature_id in 1:num_features
            d_list[id, feature_id] = shmem[(tid - 1) * num_features + feature_id]
        end
    end

    return

end

function increment_with_block_sums(d_predicateScan::CuDeviceMatrix{T}, d_blockSumScan::CuDeviceMatrix{T}, numElems::Integer, num_features::Integer) where {T}
    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    if (id <= numElems)
        for i in 1:num_features
            d_predicateScan[id, i] += d_blockSumScan[blockIdx().x, i]
        end
    end
    return
    
end

function gpu_exclusive_scan(input::CuArray{T}) where {T}

    numElems, num_features = size(input)
    gridSize = trunc(Int64, ceil(numElems/blockSize))
    block_sum = CuArray{T}(gridSize, num_features)
    num_reduction_steps = 0

    @cuda (gridSize, blockSize, shmem=blockSize * num_features * sizeof(T)) partial_exclusive_scan(input, block_sum, numElems, num_features)
    block_sum_array = CuArray{T}[]
    push!(block_sum_array, block_sum)


    while (gridSize > 1)
        new_gridSize = trunc(Int64, ceil(gridSize/blockSize))
        block_sum = CuArray{T}(new_gridSize, num_features)
        num_reduction_steps += 1
        @cuda (new_gridSize, blockSize, shmem=blockSize * num_features * sizeof(T)) partial_exclusive_scan(block_sum_array[num_reduction_steps], block_sum, gridSize, num_features)
        push!(block_sum_array, block_sum)

        gridSize = new_gridSize
    end

    while (num_reduction_steps != 0)
        if (num_reduction_steps == 1)
            numElems, num_features = size(input)
            gridSize, num_features = size(block_sum_array[num_reduction_steps])
            @cuda (gridSize, blockSize) increment_with_block_sums(input, block_sum_array[num_reduction_steps], numElems, num_features)
            break
        end
        gridSize, num_features = size(block_sum_array[num_reduction_steps])
        numElems, num_features = size(block_sum_array[num_reduction_steps-1])
        @cuda (gridSize, blockSize) increment_with_block_sums(block_sum_array[num_reduction_steps-1], block_sum_array[num_reduction_steps], numElems, num_features)
        num_reduction_steps -= 1
    end

    return block_sum_array[end]
end

# rows = 10
# cols = 1

# #@TODO Bug doesn't work with UInt64 arrays

# a = rand(Int64, rows, cols)
# a = a .% 10
# # a = [1 0 0 1 0 0 1 0]
# # a = a'


# cpu_a = copy(a)
# cpu_exclusive_scan(cpu_a)

# gpu_a = CuArray(a)
# reduction = gpu_exclusive_scan(gpu_a)

# using Base.Test
# @test cpu_a ≈ Array(gpu_a)



# FURTHER IMPROVEMENTS:
# - work efficiency
# - avoid memory bank conflcits