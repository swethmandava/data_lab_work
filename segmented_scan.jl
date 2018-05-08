# Work-inefficient inclusive scan
# - uses shared memory to reduce
#
# Based on http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
# Pkg.add("CUDAdrv")
# Pkg.add("CUDAnative")
# Pkg.update()
using CUDAdrv, CUDAnative, Base.Test
include("scan.jl")

blockSize = 2
function cpu_segmented_inclusive_scan(data::Matrix{T}, flag::Matrix{Int64}, op::Function) where {T}
    cols = size(data,2)
    for col in 1:cols
        accum = zero(T)
        rows = size(data,1)
        for row in 1:size(data,1)
            if (flag[row] == 1)
                accum = data[row, col]
            else
                accum = op(accum, data[row, col])
            end
            data[row,col] = accum
        end
    end

    return
end

function partial_segmented_inclusive_scan(d_list::CuDeviceMatrix{T}, d_block_sums::CuDeviceMatrix{T}, flag::CuDeviceMatrix{T}, 
    flag_stride::Integer, op::Function, numElems::Integer, num_features::Integer, flagElems::Integer) where {T}
    

    tid = threadIdx().x
    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x


    shmem = @cuDynamicSharedMem(T, blockDim().x * (num_features+2))
    flag_sh = blockDim().x * num_features
    flag_sh_original = blockDim().x * (num_features + 1)


    if (id > numElems)
        for feature_id in 1:num_features
            shmem[(tid - 1) * num_features + feature_id] = 0
        end

    else
        for feature_id in 1:num_features
            shmem[(tid - 1) * num_features + feature_id] = d_list[id, feature_id]
        end
    end


    if (((id * flag_stride) > flagElems) || (tid == 1))
        shmem[flag_sh + tid] = 0
        shmem[flag_sh_original + tid] = 0
    else
        f_temp = (flag[id * flag_stride] != flag[(id-1) * flag_stride])
        shmem[flag_sh + tid] = f_temp
        shmem[flag_sh_original + tid] = f_temp 
    end

    sync_threads()

    i = 2
    neighbor_offset = 1
    while (i <= blockDim().x)

        if (tid % i == 0)
            if (shmem[flag_sh + tid] == 0) 
                for feature_id in 1:num_features
                    shmem[(tid - 1) * num_features + feature_id] = op(shmem[(tid - 1) * num_features + feature_id], shmem[(tid - neighbor_offset - 1) * num_features + feature_id])
                end
                shmem[flag_sh + tid] = ((shmem[flag_sh + tid] == 1) || (shmem[flag_sh + tid - neighbor_offset] == 1))
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

                if (shmem[flag_sh_original + tid - neighbor_offset + 1] == 1)
                    shmem[(tid - 1) * num_features + feature_id] = 0
                elseif (shmem[flag_sh + tid - neighbor_offset] == 1)
                    shmem[(tid - 1) * num_features + feature_id] = old_neighbor
                else
                    shmem[(tid - 1) * num_features + feature_id] = op(old_neighbor, shmem[(tid - 1) * num_features + feature_id])
                end
            end
            shmem[flag_sh + tid - neighbor_offset] = 0
        end
        i = i >> 1
        neighbor_offset = neighbor_offset >> 1
        sync_threads()
    end

    if (id <= numElems)
        for feature_id in 1:num_features
            d_list[id, feature_id] = op(shmem[(tid - 1) * num_features + feature_id], d_list[id, feature_id])
        end
    end

    return

end

function increment_segmented_with_block_sums_inc(d_predicateScan::CuDeviceMatrix{T}, d_blockSumScan::CuDeviceMatrix{T}, flag::CuDeviceMatrix{T},
    flag_stride1::Integer, flag_stride2::Integer, op::Function, numElems::Integer, num_features::Integer, flagElems::Integer) where {T}
    
    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    id2 = blockIdx().x - 1

    if ((id <= numElems) && (blockIdx().x > 1) && (id * flag_stride1 <= flagElems) && (id2 * flag_stride2 <= flagElems) 
        && (flag[id * flag_stride1] == flag[id2 * flag_stride2]))
        for feature_id in 1:num_features
            d_predicateScan[id, feature_id] = op(d_predicateScan[id, feature_id], d_blockSumScan[id2, feature_id])
        end
    end
    return
    
end

function get_reduction(input::CuDeviceMatrix{T}, flag::CuDeviceMatrix{T}, flag_scan::CuDeviceMatrix{T}, segmented_reduction::CuDeviceMatrix{T}, 
    numElems::Integer, num_features::Integer) where {T}

    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x

    if ((id < numElems) && (flag[id] == 1))
        for feature_id in 1:num_features
            segmented_reduction[flag_scan[id-1] + 1, feature_id] = input[id - 1, feature_id]
        end
    elseif (id == numElems)
        for feature_id in 1:num_features
            segmented_reduction[flag_scan[id] + 1, feature_id] = input[id, feature_id]
        end
        if (flag[id] == 1)
            for feature_id in 1:num_features
                segmented_reduction[flag_scan[id-1] + 1, feature_id] = input[id - 1, feature_id]
            end
        end
    end

    return
end


function gpu_segmented_inclusive_scan(input::CuArray{T}, flag::CuArray{T}, op::Function) where {T}

    flag_scan = copy(flag)
    num_segments = Array(gpu_inclusive_scan(flag_scan))[1] + 1
    return gpu_segmented_inclusive_scan(input, flag, flag_scan, op, num_segments)
end



function gpu_segmented_inclusive_scan(input::CuArray{T}, flag::CuArray{T}, flag_scan::CuArray{T}, op::Function, num_segments::Int64) where {T}
    
    numElems, num_features = size(input)
    flagElems = numElems
    gridSize = trunc(Int64, ceil(numElems/blockSize))
    block_sum = CuArray{T}(gridSize, num_features)
    num_reduction_steps = 0
    flag_stride = 1

    @cuda (gridSize, blockSize, shmem = blockSize * (num_features+2) * sizeof(T)) partial_segmented_inclusive_scan(input, block_sum, flag_scan, 
        flag_stride, op, numElems, num_features, flagElems)

    block_sum_array = CuArray{T}[]
    push!(block_sum_array, block_sum)


    while (gridSize > 1)
        new_gridSize = trunc(Int64, ceil(gridSize/blockSize))
        block_sum = CuArray{T}(new_gridSize, num_features)
        num_reduction_steps += 1
        flag_stride *= blockSize
        @cuda (new_gridSize, blockSize, shmem = blockSize * (num_features+2) * sizeof(T)) partial_segmented_inclusive_scan(block_sum_array[num_reduction_steps], 
            block_sum, flag_scan, flag_stride, op, gridSize, num_features, flagElems)
        push!(block_sum_array, block_sum)

        gridSize = new_gridSize
    end

    

    while (num_reduction_steps != 0)
        if (num_reduction_steps == 1)
            numElems, num_features = size(input)
            gridSize, num_features = size(block_sum_array[num_reduction_steps])
            @cuda (gridSize, blockSize) increment_segmented_with_block_sums_inc(input, block_sum_array[num_reduction_steps], flag_scan, trunc(Int64, flag_stride/blockSize), 
                flag_stride, op, numElems, num_features, flagElems)
            break
        end
        gridSize, num_features = size(block_sum_array[num_reduction_steps])
        numElems, num_features = size(block_sum_array[num_reduction_steps-1])
        @cuda (gridSize, blockSize) increment_segmented_with_block_sums_inc(block_sum_array[num_reduction_steps-1], block_sum_array[num_reduction_steps], 
            flag_scan, trunc(Int64, flag_stride/blockSize), flag_stride, op, numElems, num_features, flagElems)
        num_reduction_steps -= 1
        flag_stride = trunc(Int64, flag_stride/blockSize)
    end

    segmented_reduction = CuArray{T}(num_segments, num_features)
    @cuda (gridSize, blockSize) get_reduction(input, flag, flag_scan, segmented_reduction, numElems, num_features)

    return segmented_reduction
end
















function partial_segmented_exclusive_scan(d_list::CuDeviceMatrix{T}, d_block_sums::CuDeviceMatrix{T}, flag::CuDeviceMatrix{T}, 
    flag_stride::Integer, op::Function, numElems::Integer, num_features::Integer, flagElems::Integer) where {T}
    

    tid = threadIdx().x
    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x


    shmem = @cuDynamicSharedMem(T, blockDim().x * (num_features+2))
    flag_sh = blockDim().x * num_features
    flag_sh_original = blockDim().x * (num_features + 1)


    if (id > numElems)
        for feature_id in 1:num_features
            shmem[(tid - 1) * num_features + feature_id] = 0
        end

    else
        for feature_id in 1:num_features
            shmem[(tid - 1) * num_features + feature_id] = d_list[id, feature_id]
        end
    end


    if (((id * flag_stride) > flagElems) || (tid == 1))
        shmem[flag_sh + tid] = 0
        shmem[flag_sh_original + tid] = 0
    else
        f_temp = (flag[id * flag_stride] != flag[(id-1) * flag_stride])
        shmem[flag_sh + tid] = f_temp
        shmem[flag_sh_original + tid] = f_temp 
    end

    sync_threads()

    i = 2
    neighbor_offset = 1
    while (i <= blockDim().x)

        if (tid % i == 0)
            if (shmem[flag_sh + tid] == 0) 
                for feature_id in 1:num_features
                    shmem[(tid - 1) * num_features + feature_id] = op(shmem[(tid - 1) * num_features + feature_id], shmem[(tid - neighbor_offset - 1) * num_features + feature_id])
                end
                shmem[flag_sh + tid] = ((shmem[flag_sh + tid] == 1) || (shmem[flag_sh + tid - neighbor_offset] == 1))
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

                if (shmem[flag_sh_original + tid - neighbor_offset + 1] == 1)
                    shmem[(tid - 1) * num_features + feature_id] = 0
                elseif (shmem[flag_sh + tid - neighbor_offset] == 1)
                    shmem[(tid - 1) * num_features + feature_id] = old_neighbor
                else
                    shmem[(tid - 1) * num_features + feature_id] = op(old_neighbor, shmem[(tid - 1) * num_features + feature_id])
                end
            end
            shmem[flag_sh + tid - neighbor_offset] = 0
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

function increment_segmented_with_block_sums(d_predicateScan::CuDeviceMatrix{T}, d_blockSumScan::CuDeviceMatrix{T}, flag::CuDeviceMatrix{T},
    flag_stride1::Integer, flag_stride2::Integer, op::Function, numElems::Integer, num_features::Integer, flagElems::Integer) where {T}
    
    id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    id2 = blockIdx().x

    if ((id <= numElems) && (blockIdx().x > 1) && (id * flag_stride1 <= flagElems) && (id2 * flag_stride2 <= flagElems) 
        && (flag[id * flag_stride1] == flag[id2 * flag_stride2]))
        for feature_id in 1:num_features
            d_predicateScan[id, feature_id] = op(d_predicateScan[id, feature_id], d_blockSumScan[id2, feature_id])
        end
    end
    return
    
end


function gpu_segmented_exclusivee_scan(input::CuArray{T}, flag::CuArray{T}, op::Function) where {T}

    flag_scan = copy(flag)
    num_segments = Array(gpu_exclusive_scan(flag_scan))[1] + 1
    return gpu_segmented_exclusive_scan(input, flag, flag_scan, op, num_segments)
end



function gpu_segmented_exclusive_scan(input::CuArray{T}, flag::CuArray{T}, flag_scan::CuArray{T}, op::Function, num_segments::Int64) where {T}
    
    numElems, num_features = size(input)
    flagElems = numElems
    gridSize = trunc(Int64, ceil(numElems/blockSize))
    block_sum = CuArray{T}(gridSize, num_features)
    num_reduction_steps = 0
    flag_stride = 1

    @cuda (gridSize, blockSize, shmem = blockSize * (num_features+2) * sizeof(T)) partial_segmented_exclusive_scan(input, block_sum, flag_scan, 
        flag_stride, op, numElems, num_features, flagElems)

    block_sum_array = CuArray{T}[]
    push!(block_sum_array, block_sum)


    while (gridSize > 1)
        new_gridSize = trunc(Int64, ceil(gridSize/blockSize))
        block_sum = CuArray{T}(new_gridSize, num_features)
        num_reduction_steps += 1
        flag_stride *= blockSize
        @cuda (new_gridSize, blockSize, shmem = blockSize * (num_features+2) * sizeof(T)) partial_segmented_exclusive_scan(block_sum_array[num_reduction_steps], 
            block_sum, flag_scan, flag_stride, op, gridSize, num_features, flagElems)
        push!(block_sum_array, block_sum)

        gridSize = new_gridSize
    end

    

    while (num_reduction_steps != 0)
        if (num_reduction_steps == 1)
            numElems, num_features = size(input)
            gridSize, num_features = size(block_sum_array[num_reduction_steps])
            @cuda (gridSize, blockSize) increment_segmented_with_block_sums(input, block_sum_array[num_reduction_steps], flag_scan, trunc(Int64, flag_stride/blockSize), 
                flag_stride, op, numElems, num_features, flagElems)
            break
        end
        gridSize, num_features = size(block_sum_array[num_reduction_steps])
        numElems, num_features = size(block_sum_array[num_reduction_steps-1])
        @cuda (gridSize, blockSize) increment_segmented_with_block_sums(block_sum_array[num_reduction_steps-1], block_sum_array[num_reduction_steps], 
            flag_scan, trunc(Int64, flag_stride/blockSize), flag_stride, op, numElems, num_features, flagElems)
        num_reduction_steps -= 1
        flag_stride = trunc(Int64, flag_stride/blockSize)
    end

    segmented_reduction = CuArray{T}(num_segments, num_features)
    @cuda (gridSize, blockSize) get_reduction(input, flag, flag_scan, segmented_reduction, numElems, num_features)

    return segmented_reduction
end

rows = 10
cols = 1

#@TODO Bug doesn't work with UInt64 arrays

# a = rand(Int64, rows, cols)
# a = a .% 10
# flag = rand(Int64, rows, cols)
# flag = flag .% 2
# flag = abs.(flag)

a = [1 2 3 4 5 6 7 8]
a = a'
flag = [0 1 0 0 0 1 0 0]
# flag = [0 0 0 0 0 0 0 0]
flag = flag'

# show(a)
# println()
# show(flag)

cpu_a = copy(a)
cpu_segmented_inclusive_scan(cpu_a, flag, +)

gpu_a = CuArray(a)
gpu_flag = CuArray(flag)
reduction = gpu_segmented_inclusive_scan(gpu_a, gpu_flag, +)

# show(flag)
# show(Array(reduction))
@test cpu_a ≈ Array(gpu_a)



# FURTHER IMPROVEMENTS:
# - work efficiency
# - avoid memory bank conflcits