#include <string.h>
#include "tree.hh"
#include <vector>
#include <algorithm>
#include <limits>

// Should be parallelized
double sum_Y(unsigned long num_features, double** X, unsigned long num)
{
    double sum = 0;

    for (unsigned long i = 0; i < num; i++)
    {
        sum += X[i][num_features];
    }

    return sum;
}

int compare_feature;
int compare(const void *aa, const void *bb)
{
    double *a = (double *)aa;
    double *b = (double *)bb;

    if (a[compare_feature] < b[compare_feature])
        return -1;
    else if (a[compare_feature] == b[compare_feature])
        return 0;
    else
        return 1;
}


void get_split(node_t* node, double lambda, double gamma)
{
    double tree_score, score;
    tree_score = - std::numeric_limits<double>::infinity();
    unsigned long tree_feature = 0;
    double tree_value = std::numeric_limits<double>::infinity();
    unsigned long num_samples = node->num_samples;
    unsigned long num_features = node->num_features;
    unsigned long tree_sample = num_samples;
    double **X = node->X;
    double G = sum_Y(num_features, X, num_samples);

    for (unsigned long i = 0; i < num_features; i++)
    {
        double Gl = 0;
        unsigned long Hl = 0;
        double Gr;
        unsigned long Hr;

        // Sort 2D array X by feature/column i
        compare_feature = i;
        std::qsort(X, num_samples, sizeof(X[0]), compare);

        // Maybe make Y a column of X only to avoid another sorting pass
        for (unsigned long j = 0; j < num_samples; j++)
        {
            Gl += X[i][num_features];
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
                tree_value = X[j][i];
                tree_sample = j;
            }
        }
    }

    if (tree_score <= 0)
    {
        node->left = nullptr;

        node->right = nullptr;

        //node->feature = tree_feature;
        //node->value = tree_value;
    }
    else
    {
        compare_feature = tree_feature;
        std::qsort(X, num_samples, sizeof(X[0]), compare);

#if 0
        node->left = new node_t;
        node->left->X = new double*[tree_sample];
        for (unsigned long i = 0; i < tree_sample; i++)
        {
            node->left->X[i] = new double[num_features + 1];
            memcpy(node->left->X[i], X[i], sizeof(double) * (num_features
                        + 1));
        }

        node->right = new node_t;
        node->right->X = new double*[num_samples - tree_sample];
        X = X + (tree_sample);
        for (unsigned long i = 0; i < num_samples - tree_sample; i++)
        {
            node->right->X[i] = new double[num_features + 1];
            memcpy(node->left->X[i], X[i], sizeof(double) * (num_features
                        + 1));
        }

        node->left->num_samples = tree_sample;
        node->right->num_samples = num_samples - tree_sample;
        node->left->num_features = num_features;
        node->right->num_features = num_features;
        node->feature = tree_feature;
        node->value = tree_value;

        for (unsigned long i = 0; i < num_samples; i++)
        {
            delete node->X[i];
        }

        delete node->X;
#endif
        node->left = new node_t;
        node->left->X = node->X;

        node->right = new node_t;
        node->right->X = node->X + tree_sample;

        node->left->num_samples = tree_sample;
        node->right->num_samples = num_samples - tree_sample;
        node->left->num_features = num_features;
        node->right->num_features = num_features;
        node->feature = tree_feature;
        node->value = tree_value;
    }

}

void terminal(node_t* node, double learning_rate)
{
    if (node != nullptr)
    {
        unsigned long div = std::max((unsigned long)1, node->num_samples);
        double sum = sum_Y(node->num_features, node->X, node->num_samples);

        node->value = (sum / div) * learning_rate;

        for (unsigned long i = 0; i < node->num_samples; i++)
        {
            delete node->X[i];
        }

        node->left = nullptr;
        node->right = nullptr;
    }
}

void split(node_t* node, unsigned long max_depth, unsigned long min_size,
           unsigned long depth, double learning_rate, double lambda, double gamma)
{
#if 0
    if (depth >= max_depth)
    {
        terminal(node->left, learning_rate);
        terminal(node->right, learning_rate);

        return;
    }

    if (node->left != nullptr)
    {
        if (node->left->num_samples <= min_size)
        {
            terminal(node->left, learning_rate);
        }
        else
        {
            get_split(node->left, lambda, gamma);
            split(node->left, max_depth, min_size, depth + 1,
                    learning_rate, lambda, gamma);
        }
    }

    if (node->right != nullptr)
    {
        if (node->right->num_samples <= min_size)
        {
            terminal(node->right, learning_rate);
        }
        else
        {
            get_split(node->right, lambda, gamma);
            split(node->right, max_depth, min_size, depth + 1,
                    learning_rate, lambda, gamma);
        }
    }
#else
    /* if (depth >= max_depth)
    {
        terminal(node, learning_rate);

        return;
    }
    */
    if (node->left == nullptr && node->right == nullptr)
    {
        terminal(node, learning_rate);
    }
    if (node->left != nullptr)
    {
        if ((node->left->num_samples <= min_size) || (depth >= max_depth))
        {
            terminal(node->left, learning_rate);
        }
        else
        {
            get_split(node->left, lambda, gamma);
            split(node->left, max_depth, min_size, depth + 1,
                    learning_rate, lambda, gamma);
        }
    }

    if (node->right != nullptr)
    {
        if ((node->right->num_samples <= min_size) || (depth >= max_depth))
        {
            terminal(node->right, learning_rate);
        }
        else
        {
            get_split(node->right, lambda, gamma);
            split(node->right, max_depth, min_size, depth + 1,
                    learning_rate, lambda, gamma);
        }
    }
#endif
}

void train (node_t* root, unsigned long max_depth, unsigned long min_size,
        double lambda, double gamma, double learning_rate)
{
    get_split(root, lambda, gamma);
    split(root, max_depth, min_size, 1, learning_rate, lambda, gamma);
}

double* predict(unsigned long num_samples, double** X, node_t* model, bool
        class_flag)
{
    double* Y = new double[num_samples];
    node_t *node;

    for (unsigned long i = 0; i < num_samples; i++)
    {
        node = model;
        while (1)
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
            else if(X[i][node->feature] < node->value)
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
