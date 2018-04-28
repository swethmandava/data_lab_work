# Simple tests for y=sin(x) for regression tree and gradient boosting tree
# Author : Swetha Mandava
# Email : mmandava@andrew.cmu.edu

Pkg.add("PyPlot")
using PyPlot
Pkg.add("DataFrames")
using DataFrames;
push!(LOAD_PATH, pwd())
include("tree.jl")
include("gradient_boosting.jl")

function print_tree(node, depth=0)
	if isa(node, Dict)
		tabs = " " ^ depth
		println(tabs, (node["index"]+1), node["value"])
		print_tree(node["left"], depth+1)
		print_tree(node["right"], depth+1)
	else
		tabs = " " ^ depth
		println(tabs, node)
	end
	return 
end

function test_regression_tree()
	x_train = collect(0: 0.1 : 100)
	n = size(x_train)[1]
	y_train = sin(x_train)
	x_train = reshape(x_train, (n, 1))

	max_depth = 20
	min_size = 2
	lambda = 0.01
	gamma = 0.01
	model = train(x_train, y_train, max_depth, min_size, lambda, gamma)

	#uncomment to print tree
	# print_tree(model)

	y_test = predict(x_train, model)
	# show(y_test)

	err = y_test - y_train
	err = sum(err.^2)/n
	show(err)
	# plot(x_train, y_train)
	# plot(x_train, y_test)
	# plt[:show]()
end

function test_gradient_boosting(filename, num_train_samples)

	max_depth = 12
	min_size = 2
	epochs = 10
	learning_rate = 1.0
	batch_size = Int(0.3 * num_train_samples)
	lambda = 0.01
	gamma = 0.01

	model = gradient_boosting(filename, num_train_samples, epochs, learning_rate, batch_size,
	max_depth, min_size, lambda, gamma)

	test = readtable(filename, skipstart=num_train_samples, nrows=Int(0.25*num_train_samples))
	num_cols = size(test)[2]
	y_test = Array(test[:, 1:1])
	x_test = Array(test[:,2:num_cols])	
	y_pred = predict_gboost(x_test, model)
	num_correct = count(i->y_test[i] == y_pred[i], [1:size(y_test)[1];])
	show("Accuracy = ")
	show((num_correct * 1.0)/size(y_test)[1])
end

# file = "Dataset/HIGGS.csv"
# data = readtable(file)
# num_samples = 11000000
# train_split = 10500000
# y_train = data[1:train_samples, 1]
# x_train = data[1:train_samples, 2:28]


# show(x_train)
# show(y_train)
# test_gradient_boosting(file, 105000)

@time test_regression_tree()