Pkg.add("PyPlot")
using PyPlot
push!(LOAD_PATH, pwd())
include("tree.jl")


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


x_train = collect(0: 0.1 : 10)
n = size(x_train)[1]
y_train = sin(x_train)
x_train = reshape(x_train, (n, 1))

max_depth = 5
min_size = 2
model = train(x_train, y_train, max_depth, min_size)

#uncomment to print tree
# print_tree(model)

y_test = predict(x_train, model)

plot(x_train, y_train)
plot(x_train, y_test)
plt[:show]()
