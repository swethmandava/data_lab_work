Pkg.add("CUDAdrv")
Pkg.add("CUDAnative")
Pkg.update()
using CUDAdrv, CUDAnative
blockSize = 2
include("scan.jl")
include("segmented_scan.jl")


function cpu_segmented_sort(data::Matrix{T}, flag::Matrix{Int64}, feature_num::Int64) where {T}
	num_elements, num_features = size(data)
	prev = 1
	for i in 1:num_elements
		if (flag[i] == 1)
			data[prev:i-1, :] = sort(data[prev:i-1, :], feature_num)
			prev = i
		end
	end
	data[prev:num_elements, :] = sort(data[prev:num_elements, :], feature_num)
	return data
end




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

function set_bit(d_predicate::CuDeviceMatrix{Int64}, numElems::Integer)
	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
	if (id > numElems)
		return
	end
	d_predicate[id] = 1;
	return 
end

# function reinterpret_kernel(input::CuDeviceMatrix{T}, Type::DataType, numElems::Integer) where {T}
# 	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
# 	if (id > numElems)
# 		return
# 	end
# 	input[id] = reinterpret(Type, input[id])
# 	return 
# end

function sign_ind(input::CuDeviceMatrix{T}, shift_index::CuDeviceMatrix{T}, flag::CuDeviceMatrix{T}, flag_scan::CuDeviceMatrix{T}, 
	numElems::Integer, feature_num::Integer) where {T}
	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x
	if (id > numElems)
		return
	elseif ((id > 1) && (flag_scan[id] == flag_scan[id-1]) && (input[id - 1, feature_num] >= 0) && (input[id, feature_num] < 0))
		shift_index[flag_scan[id] + 1] = id - 1
	elseif ((id == 1) || flag[id] == 1) && (input[id, feature_num] < 0)
		shift_index[flag_scan[id] + 1] = id - 1
	end
	return 
end

function invert_ind(input::CuDeviceMatrix{T}, output::CuDeviceMatrix{T}, shift_index::CuDeviceMatrix{T}, flag_scan::CuDeviceMatrix{T},
	numElems_scan::CuDeviceMatrix{T}, numElems::Integer, num_features::Integer) where {T}
	
	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x


	if (id > numElems)
		return
	end

	segment = flag_scan[id]
	s = shift_index[segment + 1] + 1
	offset = 0
	negElems = numElems_scan[segment + 1] - s + 1
	
	if (id >= s)
		if (segment != 0)
			offset = numElems_scan[segment]
		end
		for feature_id in 1:num_features
			output[negElems - (id - s) + offset, feature_id] = input[id, feature_id]
		end
		# neg_check[negElems - (id-s) + offset] = id
	else
		for feature_id in 1:num_features
			output[negElems + id, feature_id] = input[id, feature_id]
		end
		# neg_check[negElems + id] = id
	end
	return 
end

function scatter(input::CuDeviceMatrix{T}, output::CuDeviceMatrix{T}, d_predicateTrueScan::CuDeviceMatrix{Int64},
	d_predicateFalseScan::CuDeviceMatrix{Int64}, d_predicateFalse::CuDeviceMatrix{Int64}, 
	flag_scan::CuDeviceMatrix{T}, numTrueElems::CuDeviceMatrix{T}, numElems_scan::CuDeviceMatrix{T},
	numElems::Integer, num_features::Integer) where {T}
	
	id = blockDim().x * (blockIdx().x - 1) + threadIdx().x

	if (id > numElems)
		return
	end

	if (d_predicateFalse[id] == 1)
		newLoc = d_predicateFalseScan[id] + numTrueElems[flag_scan[id] + 1] + numElems_scan[flag_scan[id] + 1]
	else
		newLoc = d_predicateTrueScan[id] + numElems_scan[flag_scan[id] + 1]
	end

	if (newLoc <= numElems)
		for feature_id in 1:num_features
			output[newLoc, feature_id] = input[id, feature_id]
		end
	end


	return
end


function gpu_segmented_sort(d_inputVals::CuArray{T}, flag::CuArray{Int64}, feature_num::Int64) where {T}


	numElems, num_features = size(d_inputVals)
	gridSize = trunc(Int64, ceil(numElems/blockSize))


	if (typeof(T) != Int64)
		# @cuda (gridSize, blockSize) reinterpret_kernel(d_inputVals, UInt64)
		d_inputVals = CuArray(reinterpret(Int64, Array(d_inputVals)))
	end	


	flag_scan = copy(flag)
	num_segments = Array(gpu_inclusive_scan(flag_scan))[1] + 1
	op = +

	set_flag = similar(flag)
	@cuda (gridSize, blockSize) set_bit(set_flag, numElems)
	set_flag_scan = gpu_segmented_inclusive_scan(set_flag, flag, flag_scan, op, num_segments)
	
	set_flag_exc = copy(set_flag_scan)
	gpu_exclusive_scan(set_flag_scan)
	gpu_inclusive_scan(set_flag_exc)


	max_bits = sizeof(Int64) * 8 - 1
	d_predicate = CuArray{Int64}(numElems, 1)
	output_temp = CuArray{Int64}(numElems, num_features)
	

	for bit in 0:max_bits

		if ((bit + 1) % 2 == 1)
			@cuda (gridSize, blockSize) check_bit(d_inputVals, d_predicate, feature_num, bit, numElems)
		else 
			@cuda (gridSize, blockSize) check_bit(output_temp, d_predicate, feature_num, bit, numElems)
		end

		d_predicateTrueScan = copy(d_predicate)
		
		d_numPredicateTrueElements = gpu_segmented_inclusive_scan(d_predicateTrueScan, flag, flag_scan, op, num_segments)
		
		@cuda (gridSize, blockSize) flip_bit(d_predicate, numElems)
		d_predicateFalseScan = copy(d_predicate)

		gpu_segmented_inclusive_scan(d_predicateFalseScan, flag, flag_scan, op, num_segments)
		
		if ((bit + 1) % 2 == 1)
			@cuda (gridSize, blockSize) scatter(d_inputVals, output_temp, d_predicateTrueScan, d_predicateFalseScan,
				d_predicate, flag_scan, d_numPredicateTrueElements, set_flag_scan, numElems, num_features)
		else 
			@cuda (gridSize, blockSize) scatter(output_temp, d_inputVals, d_predicateTrueScan, d_predicateFalseScan,
				d_predicate, flag_scan, d_numPredicateTrueElements, set_flag_scan, numElems, num_features)
		end
	end

	# show(Array(d_inputVals))
	# println()

	if (typeof(T) != Int64)		
		shift_ind = copy(set_flag_exc)
		# neg_check = similar(d_inputVals)
		# shift_ind[1] = numElems + 1
		# shift_ind = CuArray(shift_ind)
		d_outputVals = similar(output_temp)
		@cuda (gridSize, blockSize) sign_ind(d_inputVals, shift_ind, flag, flag_scan, numElems, feature_num)#, neg_check)
		# negElems = numElems - Array(shift_ind)[1] + 1
		@cuda (gridSize, blockSize) invert_ind(d_inputVals, d_outputVals, shift_ind, flag_scan, set_flag_exc, numElems, num_features)#, neg_check)
		# @cuda (gridSize, blockSize) reinterpret_kernel(d_outputVals, T)

		# show(Array(neg_check))
		# println()
		# d_outputVals = copy(output_temp)
		d_outputVals = reinterpret(T, Array(d_outputVals))
		return d_outputVals
	end	

	return Array(d_inputVals)
	
end


rows = 15
cols = 1
a = randn(Float64, rows, cols)

# a = [-1.905 -1.70486 0.250875 0.074934 -0.00603303 0.933707 1.424 0.0546902 0.154733]
# a = a'
# flag = [0 0 0 1 0 0 1 0 0]
flag = [0 0 0 0 0 0 1 0 0 0 0 0 1 0 0]
# flag = [0 0 0 0 0 0 0 0]
flag = flag'
cpu_a = copy(a)
cpu_a = cpu_segmented_sort(cpu_a, flag, 1)
gpu_a = CuArray(a)
gpu_flag = CuArray(flag)
sorted_a = gpu_segmented_sort(gpu_a, gpu_flag, 1)

show(cpu_a)
println()
show(Array(sorted_a))
using Base.Test
@test cpu_a ≈ Array(sorted_a)
