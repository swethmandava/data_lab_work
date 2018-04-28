#include <string.h>
#include <vector>
#include <algorithm>
#include <limits>

typedef struct _node_t {
    struct _node_t* left;
    struct _node_t* right;
    unsigned long num_samples;
    unsigned long num_features;
    double** X;
    double value;
    unsigned long feature;
} node_t;


// Should be parallelized
double sum_Y(unsigned long num_features, double** X, unsigned long num)
{
    double sum = 0;

    for (unsigned long i = 0; i < num; i++)
    {
        sum += X[i][num_features + 1];
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
    double tree_value = std::numeric_limits<double>::infinity;
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
            Gl += X[i][num_features + 1];
            Hl += 1;
            Gr = G - Gl;
            Hr = num_samples - Hl;
            score = (Gl * Gl)/ std::max(1, (Hl + lambda))
                  + (Gr * Gr)/ std::max(1, (Hr + lambda))
                  - (G * G) / (H + lambda) - gamma;


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
        node->left = new node_t;
        node->left->X = new double*[num_samples];
        for (unsigned long i = 0; i < num_samples; i++)
        {
            node->left->X[i] = new double[num_features + 1];
        }

        memcpy(node->left->X, X, sizeof(double) * num_samples * (num_features + 1));
        node->right = nullptr;
        node->right->num_samples = 0;

        node->left->num_samples = num_samples;
        node->feature = tree_feature;
        node->value = tree_value;
    }
    else
    {
        compare_feature = tree_feature;
        std::qsort(X, num_samples, sizeof(X[0]), compare);

        node->left = new node_t;
        node->left->X = new double*[tree_sample - 1];
        for (unsigned long i = 0; i < tree_sample - 1; i++)
        {
            node->left->X[i] = new double[num_features + 1];
        }
        memcpy(node->left->X, X, sizeof(double) * (num_features + 1) * (tree_sample - 1));

        node->right = new node_t;
        node->right->X = new double*[num_samples - tree_sample + 1];
        for (unsigned long i = 0; i < num_samples - tree_sample + 1; i++)
        {
            node->right->X[i] = new double[num_features + 1];
        }
        memcpy(node->right->X, X + (tree_sample - 1),
                sizeof(double) * (num_features + 1) * (num_samples - tree_sample + 1));

        node->left->num_samples = tree_sample - 1;
        node->right->num_samples = num_samples - tree_sample + 1;
        node->feature = tree_feature;
        node->value = tree_value;
    }
    delete node->X;
}

void terminal(node_t* node, double learning_rate)
{
    unsigned long div = std::max(1, node->num_samples);
    double sum = sum_Y(node->num_features, node->X, node->num_samples);

    node->value = (sum / div) * learning_rate;
    delete node->X;
    node->left = nullptr;
    node->right = nullptr;
}

void split(node_t* node, unsigned long max_depth, unsigned long min_size,
           unsigned long depth, double learning_rate, double lambda, double gamma)
{
    if (depth >= max_depth)
    {
        terminal(node->left);
        terminal(node->right);

        return;
    }

    if (node->left->num_samples <= min_size)
    {
        terminal(node->left);
    }
    else
    {
        get_split(node->left, lambda, gamma);
        split(node->left, max_depth, min_size, depth + 1,
                learning_rate, lambda, gamma);
    }

    if (node->right->num_samples <= min_size)
    {
        terminal(node->right);
    }
    else
    {
        get_split(node->right, lambda, gamma);
        split(node->right, max_depth, min_size, depth + 1,
                learning_rate, lambda, gamma);
    }
}

void train (node_t* root, unsigned long max_depth, unsigned long min_size,
        double lambda, double gamma, double learning rate = 1)
{
    get_split(root, lambda, gamma);
    split(root, max_depth, min_size - 1, learning_rate, landa, gamma);
}

double* predict(unsigned long num_samples, unsigned long num_features,
            double X[][num_features], node_t* model)
{
    double* Y = new double[num_samples];

    for (unsigned long i = 0; i < num_samples, i++)
    {
        if (X[i][model->feature] < model->value)
        {
            if (model->left->left == nullptr && model->left->right == nullptr)
            {
                Y[i] = model->left->value;
            }
            else
            {
                Y[i] = predict(num_samples, num_features, X, model->left);
            }
        }
        else
        {

            if (model->right->left == nullptr && model->right->right == nullptr)
            {
                Y[i] = model->right->value;
            }
            else
            {
                Y[i] = predict(num_samples, num_features, X, model->right);
            }
        }
    }

    return Y;
}
