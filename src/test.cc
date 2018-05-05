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
    root->num_samples = 1000;
    root->num_features = 1;
    root->X = new double*[root->num_samples];
    double** X = new double*[root->num_samples];
    double count = 0;
    for (unsigned long i = 0; i < root->num_samples; i++, count += 0.1)
    {
        root->X[i] = new double[root->num_features + 1];
        root->X[i][0] = count;
        root->X[i][1] = std::sin(count);
        X[i] = new double[root->num_features + 1];
        X[i][0] = count;
        X[i][1] = std::sin(count);
    }

    unsigned long max_depth = 20;
    unsigned long min_size = 2;
    double lambda = 0.01;
    double gamma = 0.01;
    double learning_rate = 1;

    train(root, max_depth, min_size, lambda, gamma, learning_rate);
    int depth = 0;
    //print_tree(root, depth);

    double *Y = predict(root->num_samples, X, root, false);

#if 1
    count = 0;
    double mse = 0;
    double err;
    for (unsigned long i = 0; i < root->num_samples; i++)
    {
        //printf("%lf, %lf\n", X[i][0], Y[i]);
        err = (X[i][1] - Y[i]);
        err *= err;
        mse += err;
    }
    printf("MSE: %lf\n", mse/root->num_samples);
#endif
    delete Y;
    for (unsigned long i = 0; i < root->num_samples; i++, count += 0.1)
    {
        delete X[i];
    }
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

    unsigned long batch_size = (unsigned long)(0.3 * num_samples);
    node_t** model = gradient_boosting(filename, num_samples, epochs,
                                    learning_rate, max_depth, min_size,
                                    lambda, gamma, batch_size, num_features);

    node_t* root = new node_t;
    root->num_samples = (unsigned long) (0.3 * num_samples);
    root->num_features = 28;
    std::ifstream f;
    f.open(filename);

    readcsv(f, num_samples, root);
    double* Y_predict = predict_gboost(root, model, epochs);

    unsigned long num_correct = 0;

    for (unsigned long i = 0; i < root->num_samples; i++)
    {
        if (root->X[i][root->num_features] == Y_predict[i])
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
    //test_regression_tree();
    test_gradient_boosting();
    auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;

	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}
