function test_split(index, value, X, Y)
	n, m = size(X)
	left, right = Array{Float64}(0, m), Array{Float64}(0, m)
	left_y, right_y = Array{Float64}(0), Array{Float64}(0)
	for i in 1:n
		if X[i, index] < value
			left = [left;  X[i, :]']
			left_y = [left_y; Y[i]]
		else
			right = [right; X[i, :]']
			right_y = [right_y; Y[i]]
		end
	end
	return left, left_y, right, right_y
end

function cart_score(groups)
	score = 0.0
	for group in groups
		ni = size(group)[1]
		if ni == 0
			continue
		end
		mean = sum(group, 1)/ni
		score_i = broadcast(-, group, mean)
		score += sum(score_i.*score_i)
	end

	return score
end

function get_split(X, Y)
	n, m = size(X)
	tree_index, tree_value, tree_score, tree_groups = Inf, Inf, Inf, Inf, nothing
	for index in 1:m
		for i in 1:n
			left, left_y, right, right_y = test_split(index, X[i][index], X, Y)
			score = cart_score((left_y, right_y))
			if score < tree_score
				tree_index, tree_value, tree_score, tree_groups = index, X[i][index], score, ((left, left_y), (right, right_y))
			end
		end
	end

	return Dict("index"=>tree_index, "value"=>tree_value, "groups"=>tree_groups)
end

function split(node, max_depth, min_size, depth)
	(left, left_y), (right, right_y) = node["groups"]
	nl, nm = size(left)
	nr, nm = size(right)
	delete!(node, "groups")

	if depth >= max_depth
		node["left"] = to_terminal(left_y)
		node["right"] = to_terminal(right_y)
		return node
	end

	if nl <= min_size
		node["left"] = to_terminal(left_y)
	else 
		node["left"] = get_split(left, left_y)
		node["left"] = split(node["left"], max_depth, min_size, depth+1)
	end
	if nr <= min_size
		node["right"] = to_terminal(right_y)
	else 
		node["right"] = get_split(right, right_y)
		node["right"] = split(node["right"], max_depth, min_size, depth+1)
	end
	return node
end

function to_terminal(group)
	ni = size(group)[1]
	mean = sum(group, 1)/ni
	return mean
end

function train(X, Y, max_depth, min_size)
	root = get_split(X, Y)
	model = split(root, max_depth, min_size, 1)
	return model
end

function predict(X, model)
	n, m = size(X)
	Y = Array{Float64}(0)
	for i = 1: n 
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
