# Gradient Boosting Algorithm
# Presently supports only regression trees of max depth and minimum size of leaves
# Presently supports only square loss
# Author : Swetha Mandava
# Email : mmandava@andrew.cmu.edu

include("tree.jl")

function gradient_boosting(filename, num_samples::Int64, epochs::Int64, learning_rate::Float64, batch_size::Int64,
	max_depth::Int64, min_size::Int64, lambda::Float64, gamma::Float64)

	F0 = 0
	model = dummy_tree(F0)
	weak_learners = Dict(0=>model)
	# Y_residue = broadcast(-, Y, F0)
	# n,m = size(X)
	# permutation = randperm(n)

	for i in 0:epochs
		# X = X[permutation, :]
		# Y_residue = Y_residue[permutation, :]
		show("Iteration = ")
		show(i)
		start_rand_index = rand(1:num_samples)
		batch = readtable(filename, skipstart=start_rand_index, nrows=batch_size)
		num_cols = size(batch)[2]
		batch_x = Array(batch[:, 2:num_cols])
		batch_y = Array(batch[:, 1:1])

		batch_y_residue = batch_y - predict_gboost(batch_x, weak_learners)
		model = train(batch_x, batch_y_residue, max_depth, min_size, lambda, gamma, learning_rate)
		weak_learners[i] = model
		
	end
	return weak_learners
end

function predict_gboost(X, weak_learners)
	n,m = size(X)
	Y = Array{Float64}(0, 0)
	for (model_no, model) in weak_learners
		try
			Y += predict(X, model)
		catch
			Y = predict(X, model)
		end
	end
	return Y
end

