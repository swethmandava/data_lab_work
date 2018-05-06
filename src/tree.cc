#include <string.h>
#include "tree.hh"
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

// Should be parallelized
double sum_Y(unsigned long num_features, double* X, unsigned long num, unsigned
        long cols)
{
    double sum = 0;

    for (unsigned long i = 0; i < num; i++)
    {
        sum += X[i * cols + num_features];
    }

    return sum;
}

int compare_feature;
int compare(const void *aa, const void *bb)
{
    double *a = (double *)aa;
    double *b = (double *)bb;

    double a_num = a[compare_feature];
    double b_num = b[compare_feature];
    if (a_num < b_num)
        return -1;
    else if (a_num == b_num)
        return 0;
    else
        return 1;
}


void get_split(node_t* node, double lambda, double gamma, unsigned long cols)
{
    double tree_score, score;
    tree_score = - std::numeric_limits<double>::infinity();
    unsigned long tree_feature = 0;
    double tree_value = std::numeric_limits<double>::infinity();
    unsigned long num_samples = node->num_samples;
    unsigned long num_features = node->num_features;
    unsigned long tree_sample = num_samples;
    double *X = node->X;
    double G = sum_Y(num_features, X, num_samples, cols);

    for (unsigned long i = 0; i < num_features; i++)
    {
        double Gl = 0;
        unsigned long Hl = 0;
        double Gr;
        unsigned long Hr;

        // Sort 2D array X by feature/column i
        compare_feature = i;
        std::qsort(X, num_samples, sizeof(double) * cols, compare);

        for (unsigned long j = 0; j < num_samples; j++)
        {
            Gl += X[j * cols + num_features];
            Hl += 1;
            Gr = G - Gl;
            Hr = num_samples - Hl;

            score = (Gl * Gl)/ std::max((double)1, (Hl + lambda))
                  + (Gr * Gr)/ std::max((double)1, (Hr + lambda))
                  - (G * G) / (num_samples + lambda) - gamma;

            if (score > tree_score)
            {
                tree_score = score;
                tree_feature = i;
                tree_value = X[j * cols + i];
                tree_sample = j;
            }
        }
    }

    if (tree_score <= 0)
    {
        node->left = nullptr;

        node->right = nullptr;

    }
    else
    {
        compare_feature = tree_feature;
        std::qsort(X, num_samples, sizeof(double) * cols, compare);

        node->left = new node_t;
        node->left->X = X;

        node->right = new node_t;
        node->right->X = X + (tree_sample * cols);

        node->left->num_samples = tree_sample;
        node->right->num_samples = num_samples - tree_sample;
        node->left->num_features = num_features;
        node->right->num_features = num_features;
        node->feature = tree_feature;
        node->value = tree_value;
    }

}

void terminal(node_t* node, double learning_rate, unsigned long cols)
{
    unsigned long num_samples = node->num_samples;
    double* X = node->X;
    if (node != nullptr)
    {
        unsigned long div = std::max((unsigned long)1, num_samples);
        double sum = sum_Y(node->num_features, X, num_samples, cols);

        node->value = (sum / div) * learning_rate;

        for (unsigned long i = 0; i < num_samples; i++)
        {
            //delete X[i];
        }

        node->left = nullptr;
        node->right = nullptr;
    }
}

void split(node_t* node, unsigned long max_depth, unsigned long min_size,
           unsigned long depth, double learning_rate, double lambda, double gamma, unsigned long cols)
{
    if (node->left == nullptr && node->right == nullptr)
    {
        terminal(node, learning_rate, cols);
    }
    if (node->left != nullptr)
    {
        if ((node->left->num_samples <= min_size) || (depth >= max_depth))
        {
            terminal(node->left, learning_rate, cols);
        }
        else
        {
            get_split(node->left, lambda, gamma, cols);
            split(node->left, max_depth, min_size, depth + 1,
                    learning_rate, lambda, gamma, cols);
        }
    }

    if (node->right != nullptr)
    {
        if ((node->right->num_samples <= min_size) || (depth >= max_depth))
        {
            terminal(node->right, learning_rate, cols);
        }
        else
        {
            get_split(node->right, lambda, gamma, cols);
            split(node->right, max_depth, min_size, depth + 1,
                    learning_rate, lambda, gamma, cols);
        }
    }
}

void train (node_t* root, unsigned long max_depth, unsigned long min_size,
        double lambda, double gamma, double learning_rate, unsigned long cols)
{
    get_split(root, lambda, gamma, cols);
    split(root, max_depth, min_size, 1, learning_rate, lambda, gamma, cols);
}

double* predict(unsigned long num_samples, double* X, node_t* model, bool
        class_flag, unsigned long cols)
{
    double* Y = new double[num_samples];
    node_t *node;

    for (unsigned long i = 0; i < num_samples; i++)
    {
        node = model;
        while (node)
        {
            if (node->left == nullptr && node->right == nullptr)
            {
                if (class_flag)
                {
                    if (node->value >= 0.5)
                        Y[i] = 1;
                    else
                        Y[i] = 0;
                }
                else
                {
                    Y[i] = node->value;
                }
                break;
            }
            else if(X[i * cols + node->feature] < node->value)
            {
                node = node->left;
            }
            else
            {
                node = node->right;
            }
        }
    }

    return Y;
}
