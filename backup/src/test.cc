#include "tree.hh"
#include "gradient_boosting.hh"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>

void print_tree(node_t* root, int depth)
{
    if (root == nullptr)
        return;

    printf("depth = %d, feature = %ld, value = %lf\n", depth, root->feature,
            root->value);
    depth = depth + 1;

    print_tree(root->left, depth);
    print_tree(root->right, depth);

}

void test_regression_tree()
{
    node_t* root = new node_t;
    root->num_samples = 10000000;
    root->num_features = 1;
    unsigned long num_samples = root->num_samples;
    unsigned long num_features = root->num_features;
    unsigned long cols = num_features + 1;
    root->X = new double[num_samples * cols];
    root->X_index = 0;
    double* X = new double[num_samples * cols];

    double count = 0;
    for (unsigned long i = 0; i < root->num_samples; i++, count += 0.1)
    {
        root->X[i * cols + 0] = count;
        root->X[i * cols + 1] = std::sin(count);
        X[i * cols + 0] = count;
        X[i * cols + 1] = std::sin(count);
    }

    unsigned long max_depth = 20;
    unsigned long min_size = 2;
    double lambda = 0.01;
    double gamma = 0.01;
    double learning_rate = 1;

    train(root, max_depth, min_size, lambda, gamma, learning_rate, cols);
    int depth = 0;
    //print_tree(root, depth);

    double *Y = predict(num_samples, X, root, false, cols);

#if 1
    count = 0;
    double mse = 0;
    double err;
    for (unsigned long i = 0; i < num_samples; i++)
    {
        err = (X[i * cols + 1] - Y[i]);
        err *= err;
        mse += err;
    }
    printf("MSE: %lf\n", mse/num_samples);
#endif
    delete Y;
    delete X;
}

void test_gradient_boosting()
{
    unsigned long max_depth = 12;
    unsigned long min_size = 2;
    unsigned long epochs = 100;
    unsigned long learning_rate = 0.1;
    double lambda = 0.01;
    double gamma = 0.01;

    char* filename = "HIGGS.csv";
    unsigned long num_samples = (unsigned long) (0.7 * 11000000);
    unsigned long num_features = 28;
    unsigned long cols = num_features + 1;

    unsigned long batch_size = (unsigned long)(0.3 * num_samples);
    node_t** model = gradient_boosting(filename, num_samples, epochs,
                                    learning_rate, max_depth, min_size,
                                    lambda, gamma, batch_size, num_features,
                                    cols);

    node_t* root = new node_t;
    root->num_samples = (unsigned long) (0.3 * num_samples);
    root->num_features = 28;
    std::ifstream f;
    f.open(filename);

    readcsv(f, num_samples, root, cols);
    double* Y_predict = predict_gboost(root, model, epochs, cols);

    unsigned long num_correct = 0;

    for (unsigned long i = 0; i < root->num_samples; i++)
    {
        if (root->X[i * cols + root->num_features] == Y_predict[i])
            num_correct++;
    }

    std::cout << "accuracy = "<< ((double) (num_correct))/ root->num_samples <<
        std::endl;

    f.close();
    delete Y_predict;
}

int main()
{
	auto start = std::chrono::system_clock::now();
    test_regression_tree();
    //test_gradient_boosting();
    auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
