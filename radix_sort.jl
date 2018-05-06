Pkg.add("CUDAdrv")
Pkg.add("CUDAnative")
Pkg.update()
using CUDAdrv, CUDAnative
blockSize = 1024
include("scan.jl")

function check_bit(d_inputVals::CuDeviceMatrix{T}, d_predicate::CuDeviceMatrix{Int64}, feature_num::Integer, bit::Integer, numElems::Integer) where {T}

	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
	if (id > numElems)
		return
	end

	d_predicate[id] = (((d_inputVals[id, feature_num] >> bit) & 1) == 0)
	return
end

function flip_bit(d_predicate::CuDeviceMatrix{Int64}, numElems::Integer)
	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
	if (id > numElems)
		return
	end
	d_predicate[id] = ((d_predicate[id] + 1) % 2);
	return 
end

function scatter(input::CuDeviceMatrix{T}, output::CuDeviceMatrix{T}, d_predicateTrueScan::CuDeviceMatrix{Int64},
	d_predicateFalseScan::CuDeviceMatrix{Int64}, d_predicateFalse::CuDeviceMatrix{Int64}, numTrueElems::Integer,
	numElems::Integer, num_features::Integer) where {T}
	
	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x

	if (id > numElems)
		return
	end

	if (d_predicateFalse[id] == 1)
		newLoc = d_predicateFalseScan[id] + numTrueElems + 1
	else
		newLoc = d_predicateTrueScan[id] + 1
	end

	if (newLoc <= numElems)
		for feature_id in 1:num_features
			output[newLoc, feature_id] = input[id, feature_id]
		end
	end


	return
end


function gpu_sort(d_inputVals::CuArray{T}, feature_num::Int64) where {T}

	numElems, num_features = size(d_inputVals)
	gridSize = trunc(Int64, ceil(numElems/blockSize))

	max_bits = sizeof(T) * 8 - 1
	d_predicate = CuArray{Int64}(numElems, 1)
	d_outputVals = CuArray{T}(numElems, num_features)

	for bit in 0:max_bits

		if ((bit + 1) % 2 == 1)
			@cuda (gridSize, blockSize) check_bit(d_inputVals, d_predicate, feature_num, bit, numElems)
		else 
			@cuda (gridSize, blockSize) check_bit(d_outputVals, d_predicate, feature_num, bit, numElems)
		end

		d_predicateTrueScan = copy(d_predicate)
		
		d_numPredicateTrueElements = Array(gpu_exclusive_scan(d_predicateTrueScan))[1]

		@cuda (gridSize, blockSize) flip_bit(d_predicate, numElems)
		d_predicateFalseScan = copy(d_predicate)

		gpu_exclusive_scan(d_predicateFalseScan)

		if ((bit + 1) % 2 == 1)
			@cuda (gridSize, blockSize) scatter(d_inputVals, d_outputVals, d_predicateTrueScan, d_predicateFalseScan,
				d_predicate, d_numPredicateTrueElements, numElems, num_features)
		else 
			@cuda (gridSize, blockSize) scatter(d_outputVals, d_inputVals, d_predicateTrueScan, d_predicateFalseScan,
				d_predicate, d_numPredicateTrueElements, numElems, num_features)
		end

	end
	return Array(d_outputVals)
end


rows = 8
cols = 1

a = rand(UInt64, rows, cols)
a = a .% 10
# a = [1 2  4 5 6 2 9 0]
# a = a'
cpu_a = sort(a, 1)

gpu_a = CuArray(a)
sorted_a = gpu_sort(gpu_a, 1)

using Base.Test
@test cpu_a ≈ sorted_a