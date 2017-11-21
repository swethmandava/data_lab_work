# Gradient Boosting Algorithm
# Presently supports only regression trees of max depth and minimum size of leaves
# Presently supports only square loss
# Author : Swetha Mandava
# Email : mmandava@andrew.cmu.edu

include("tree.jl")

function gradient_boosting(X, Y, epochs, learning_rate, batch_size,
	max_depth, min_size)

	F0 = to_terminal(Y, learning_rate)
	model = dummy_tree(F0)
	weak_learners = Dict(0=>model)
	Y_residue = broadcast(-, Y, F0)
	n,m = size(X)
	permutation = randperm(n)

	for i in 1:epochs
		X = X[permutation, :]
		Y_residue = Y_residue[permutation, :]

		batch_x = X[1:batch_size, :]
		batch_y = Y_residue[1:batch_size, :]

		model = train(batch_x, batch_y, max_depth, min_size, learning_rate)
		weak_learners[i] = model
		Y_residue -= predict(X, model)
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

