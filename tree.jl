# Regression Trees Using Square Loss
# Author : Swetha Mandava
# Email : mmandava@andrew.cmu.edu

function test_split(index::Int64, value, X, Y, lamda::Float64, gamma::Float64, num_samples::Int64, num_features::Int64)
	left, right = Array{Float64}(0, num_features), Array{Float64}(0, num_features)
	left_y, right_y = Array{Float64}(0), Array{Float64}(0)
	G = sum(Y)
	Gl = 0
	H = num_samples
	Hl = 0
	for i in 1:num_samples
		if X[i, index] < value
			left = [left;  X[i, :]']
			left_y = [left_y; Y[i]]
			Gl += Y[i]
			Hl += 1
		else
			right = [right; X[i, :]']
			right_y = [right_y; Y[i]]
		end
	end
	Gr = G - Gl
	Hr = H - Hl
	score = (Gl * Gl) / max(1, (Hl + lamda)) + (Gr * Gr) / max(1, (Hr + lamda)) - (G*G)/(H + lamda) - gamma
	return left, left_y, right, right_y, score
end

function get_split(X, Y, lambda, gamma)
	num_samples, num_features = size(X)
	tree_index, tree_value, tree_score, tree_groups = Inf, Inf, -Inf, nothing
	for i in 1:num_samples
		for index in 1:num_features
			left, left_y, right, right_y, score = test_split(index, X[i][index], X, Y, 
				lambda, gamma, num_samples, num_features)

			if score > tree_score
				tree_index, tree_value, tree_score, tree_groups = index, X[i][index], score, ((left, left_y), (right, right_y))
			end
		end
	end

	if tree_score <= 0
		# Don't split i.e put all in the left tree
		left = [left; right]
		left_y = [left_y; right_y]

		#Dummy values. Should never branch to this!
		right = zeros(Float64, (1, num_features))
		right_y = zeros(Float64, 1)
		return Dict("index"=>1, "value"=>Inf, "groups"=>((left, left_y), (right, right_y)))
	else
		return Dict("index"=>tree_index, "value"=>tree_value, "groups"=>tree_groups)
	end
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
	mean = sum(group_y, 1)/num_samples
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
