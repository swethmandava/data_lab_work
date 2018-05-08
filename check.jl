# Regression Trees Using Square Loss
# Author : Swetha Mandava
# Email : mmandava@andrew.cmu.edu

# type Bst 
#     val::Int 
#     left::Nullable{Bst} 
#     right::Nullable{Bst} 
# end 
# Bst(key::Int) = Bst(key, Nullable{Bst}(), Nullable{Bst}())   


type node_t
	value
	feature::Int64
	num_samples::Int64
	start_index::Int64
	left::Nullable{node_t}
	right::Nullable{node_t}
end

node_t(value, feature::Int64, num_samples::Int64, start_index::Int64) = node_t(value, 
	feature, num_samples, start_index, Nullable{node_t}(), Nullable{node_t}())


function get_split(root, X_all, Y_all, lambda, gamma, learning_rate)

	if (isa(root, Nullable))
		root = root.value
	end
	X = X_all[root.start_index : root.start_index + root.num_samples-1, :]
	Y = Y_all[root.start_index : root.start_index + root.num_samples-1, :]
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
				tree_value = X[i, index]
				tree_sample = i
			end
		end
	end

	if (tree_score <= 0)
		root.value = to_terminal(X_all, Y_all, root, learning_rate)
	else
		sorted_x_order = sort!([1:num_samples;], by=i->X[i, tree_index])
		X_all[root.start_index : root.start_index+num_samples-1, :] = X[sorted_x_order, :]
		Y_all[root.start_index : root.start_index+num_samples-1, :] = Y[sorted_x_order, :]
		root.feature = tree_index
		root.value = tree_value
		root.left = node_t(Inf, num_features, tree_sample+1, root.start_index)
		root.right = node_t(Inf, num_features, num_samples - tree_sample - 1, root.start_index + tree_sample+1)
	end

	return
end

function split(node, X, Y, max_depth::Int64, min_size::Int64, depth::Int64, 
	learning_rate::Float64, lambda::Float64, gamma::Float64)
	
	if (isa(node, Nullable))
		node = node.value
	end
	
	if (isnull(node.left) && isnull(node.right))
		to_terminal(X, Y, node, learning_rate)
		return 
	end
	if (!isnull(node.left))
		if ((node.left.value.num_samples <= min_size) || (depth >= max_depth))
			to_terminal(X, Y, node.left, learning_rate)
		else
			get_split(node.left, X, Y, lambda, gamma, learning_rate)
			split(node.left, X, Y, max_depth, min_size, depth+1, learning_rate, lambda, gamma)
		end 
	end
	if (!isnull(node.right))
		if ((node.right.num_samples <= min_size) || (depth >= max_depth))
			to_terminal(X, Y, node.right, learning_rate)
		else
			get_split(node.right, X, Y, lambda, gamma, learning_rate)
			split(node.right, X, Y, max_depth, min_size, depth+1, learning_rate, lambda, gamma)
		end 
	end
end

function to_terminal(X, Y, node, learning_rate::Float64)
	if (isa(node, Nullable))
		node = node.value
	end
	num_samples = node.num_samples
	mean = sum(Y[node.start_index:node.start_index + node.num_samples - 1, :], 1)/max(1, num_samples)
	node.value = mean * learning_rate
	return
end

function train(X, Y, max_depth::Int64, min_size::Int64, lambda::Float64, gamma::Float64, learning_rate::Float64=1.0)
	num_samples, num_features = size(X)
	root = node_t(Inf, num_features, num_samples, 1)
	get_split(root, X, Y, lambda, gamma, learning_rate)
	split(root, X, Y, max_depth, min_size, 1, learning_rate, lambda, gamma)
	return root
end

function predict(X, model, classification_flag=false)
	num_samples, num_features = size(X)
	Y = Array{Float64}(0)
	if (isnull(model))
		return Y
	end
	for i = 1: num_samples
		node = model
		while (true)
			show(model)
			if (isnull(node.left) && isnull(node.right))
				if classification_flag
					if (node.value >= 0.5)
						Y = [Y; 1]
					else
						Y = [Y; 0]
					end
				else
					Y = [Y; node.value]
				end
				break
			elseif (X[i, node.feature] < node.value)
				node = node.left.value
			else
				node = node.right.value
			end
		end
	end
	return Y
end

#dummy tree that always returns given value
function dummy_tree(value)

	if isa(value, Array)
		one, num_features = size(value)
	else
		num_features = 1
	end
	model = node_t(value, num_features, num_samples, 1)
	return model
end
