type Bst 
    val::Int 
    left::Nullable{Bst} 
    right::Nullable{Bst} 
end 
Bst(key::Int) = Bst(key, Nullable{Bst}(), Nullable{Bst}())   

"Given an array of Ints, it will create a BST tree, type: Bst" 
function build_bst(list::Array{Int,1}) 
    head = list[1] 
    tree = Bst(head) 
    for e in list[2:end] 
        place_bst(tree,e) 
    end 
    return tree 
end 

function place_bst(tree::Bst,e::Int) 
    if e == tree.val 
        println("Dropping $(e). No repeated values allowed") 
    elseif e < tree.val 
        if (isnull(tree.left)) 
            tree.left = Bst(e) 
        else 
            place_bst(tree.left.value,e) 
        end 
    else 
        if (isnull(tree.right)) 
            tree.right = Bst(e) 
        else 
            place_bst(tree.right.value,e) 
        end 
    end 
end 

function print_bst(tree::Bst) 
	show(tree)
    if !isnull(tree.left) print_bst(tree.left.value) end 
    println(tree.val) 
    if !isnull(tree.right) print_bst(tree.right.value) end 
end 


print_bst(build_bst([4,5,10,3,20,-1,10])) 