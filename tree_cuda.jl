# Regression Trees Using Square Loss
# Author : Swetha Mandava
# Email : mmandava@andrew.cmu.edu


function get_split(X, Y, lambda, gamma)
	num_samples, num_features = size(X)

	#Default values!
	tree_index, tree_value, tree_score, tree_sample, tree_groups = num_features, Inf, -Inf, num_samples, nothing

	G = sum(Y)
	H = num_samples
	for index in 1:num_features
		Gl = 0
		Hl = 0
		sorted_x_order = sort!([1:num_samples;], by=i->X[i, index])
		X = X[sorted_x_order, :]
		Y = Y[sorted_x_order, :]
		for i in 1:num_samples
			Gl += Y[i]
			Hl += 1
			Gr = G - Gl
			Hr = H - Hl

			score = (Gl * Gl) / max(1, (Hl + lambda)) + (Gr * Gr) / max(1, (Hr + lambda)) - (G*G)/(H + lambda) - gamma
			if score > tree_score
				tree_score = score
				tree_index = index
				tree_value = X[i][index]
				tree_sample = i
			end
		end
	end

	sorted_x_order = sort!([1:num_samples;], by=i->X[i, tree_index])
	X = X[sorted_x_order, :]
	Y = Y[sorted_x_order, :]
	left = X[1:tree_sample, :]
	left_y = Y[1:tree_sample, :]
	right = X[tree_sample+1:num_samples, :]
	right_y = Y[tree_sample+1:num_samples, :]
	tree_groups = ((left, left_y), (right, right_y))
	return Dict("index"=>tree_index, "value"=>tree_value, "groups"=>tree_groups)
end

function split(node, max_depth::Int64, min_size::Int64, depth::Int64, 
	learning_rate::Float64, lambda::Float64, gamma::Float64)
	(left, left_y), (right, right_y) = node["groups"]
	left_num_samples, num_features = size(left)
	right_num_samples, num_features = size(right)
	delete!(node, "groups")

	if depth >= max_depth
		node["left"] = to_terminal(left_y, learning_rate)
		node["right"] = to_terminal(right_y, learning_rate)
		return node
	end

	if left_num_samples <= min_size
		node["left"] = to_terminal(left_y, learning_rate)
	else 
		node["left"] = get_split(left, left_y, lambda, gamma)
		node["left"] = split(node["left"], max_depth, min_size,
			depth+1, learning_rate, lambda, gamma)
	end
	if right_num_samples <= min_size
		node["right"] = to_terminal(right_y, learning_rate)
	else 
		node["right"] = get_split(right, right_y, lambda, gamma)
		node["right"] = split(node["right"], max_depth, min_size,
			depth+1, learning_rate, lambda, gamma)
	end
	return node
end

function to_terminal(group_y, learning_rate::Float64)
	num_samples = size(group_y)[1]
	mean = sum(group_y, 1)/max(1, num_samples)
	return mean * learning_rate
end

function train(X, Y, max_depth::Int64, min_size::Int64, lambda::Float64, gamma::Float64, learning_rate::Float64=1)
	root = get_split(X, Y, lambda, gamma)
	model = split(root, max_depth, min_size, 1, learning_rate, lambda, gamma)
	return model
end

function predict(X, model)
	num_samples, num_features = size(X)
	Y = Array{Float64}(0)
	for i = 1: num_samples
		if X[i, model["index"]] < model["value"]
			if isa(model["left"], Dict)
				Y = [Y; predict(X[i, :]', model["left"])]
			else
				Y = [Y; model["left"]]
			end
		else
			if isa(model["right"], Dict)
				Y = [Y; predict(X[i,:]', model["right"])]
			else
				Y = [Y; model["right"]]
			end
		end
	end
	return Y
end

#dummy tree that always returns given value
function dummy_tree(value)
	model = Dict("index"=>1, "value"=>Inf, "left"=>value)
	return model
end
