#include "tree.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string.h>
#include <string>

double* predict_gboost(node_t* root, node_t** weak_learners, unsigned long num,
        unsigned long cols)
{
    unsigned long num_samples = root->num_samples;
    if (num == 0)
    {
        return predict(num_samples, root->X, weak_learners[0], true, cols);
    }

    double* Y = new double[num_samples];
    memset(Y, 0, sizeof(double)*num_samples);
    double* Y_temp;
    for (unsigned long i = 0; i < num; i++)
    {
        Y_temp = predict(num_samples, root->X, weak_learners[i],
                true, cols);
        for (unsigned long j = 0; j < num_samples; j++)
        {
            Y[j] += Y_temp[j];
        }
        delete Y_temp;
    }
    return Y;
}

void readcsv(std::ifstream& file, unsigned long index, node_t* root, unsigned
            long cols)
{
    for (unsigned long i = 0; i < index; i++)
    {
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }

    std::string line, val;

    unsigned long num_samples = root->num_samples;
    unsigned long num_features = root->num_features;
    double* X = new double[num_samples * cols];

    for (unsigned long i = 0; i < num_samples; i++)
    {
        std::getline(file, line);
        std::stringstream s(line);
        std::getline(s, val, ',');
        X[i * cols + num_features] = std::stod(val);
        for (unsigned long  j = 0; j < num_features; j++)
        {
            std::getline(s, val, ',');
            X[i * cols + j] = std::stod(val);
        }
    }

    root->X = X;
    file.clear();
    file.seekg(0, std::ios::beg);
}

node_t** gradient_boosting(char* filename, unsigned long num_samples,
                       unsigned long epochs, double learning_rate,
                       unsigned long max_depth, unsigned long min_size,
                       double lambda, double gamma,
                       unsigned long batch_size, unsigned long num_features,
                       unsigned long cols)
{
    node_t** weak_learners = new node_t*[epochs];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long> dis(0, num_samples - batch_size);

    std::ifstream f;
    f.open(filename, std::ifstream::in);

    if (f.is_open())
    {
        std::cout << "okay" << std::endl;
    }
    else
    {
        std::cout << "bad file" << std::endl;
        return NULL;
    }
    std::string ss;

    unsigned long index;
    for (unsigned long i = 0; i < epochs; i++)
    {
        std::cout << "iteration "<< i << " " << std::endl;
        index = dis(gen);
        node_t* root = new node_t;
        root->num_samples = batch_size;
        root->num_features = num_features;
        readcsv(f, index, root, cols);

        double* Y_predict = new double[batch_size];
        Y_predict = predict_gboost(root, weak_learners, i, cols);

        for (unsigned long i = 0; i < batch_size; i++)
        {
            root->X[i * cols + num_features] = root->X[i * cols + num_features] - Y_predict[i];
        }
        delete Y_predict;
        train(root, max_depth, min_size, lambda, gamma, learning_rate, cols);
        weak_learners[i] = root;
    }
    f.close();

    return weak_learners;
}
