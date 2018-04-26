#include <string.h>
#include <vector>
#include <algorithm>
#include <limits>

typedef struct _node_t {
    double** left;
    unsigned long left_samples;
    unsigned long right_samples;
    double** right;
    double value;
    unsigned long feature;
} node_t;


// Should be parallelized
double sum(unsigned long num_features, double X[][num_features], unsigned long num)
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

void get_split(node_t* node, unsigned long num_features, unsigned long num_samples,
               double X[][num_features + 1], double Y[],
               double lambda, double gamma)
{
    double tree_score, score;
    tree_score = - std::numeric_limits<double>::infinity();
    unsigned long tree_feature = 0;
    double tree_value = std::numeric_limits<double>::infinity;
    unsigned long tree_sample = num_samples;
    double G = sum(num_features, X, num_samples);

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
        double **left = new double*[num_samples];
        for (unsigned long i = 0; i < num_samples; i++)
        {
            left[i] = new double[num_features + 1];
        }

        memcpy(left, X, sizeof(double) * num_samples * (num_features + 1));
        double **right = nullptr;

        node->left = left;
        node->left_samples = num_samples;
        node->right = right;
        node->right_samples = 0;
        node->feature = tree_feature;
        node->value = tree_value;
    }
    else
    {
        compare_feature = tree_feature;
        std::qsort(X, num_samples, sizeof(X[0]), compare);

        double **left = new double*[tree_sample - 1];
        for (unsigned long i = 0; i < tree_sample - 1; i++)
        {
            left[i] = new double[num_features + 1];
        }
        memcpy(left, X, sizeof(double) * (num_features + 1) * (tree_sample - 1));

        double **right = new double*[num_samples - tree_sample + 1];
        for (unsigned long i = 0; i < num_samples - tree_sample + 1; i++)
        {
            right[i] = new double[num_features + 1];
        }
        memcpy(right, X + (tree_sample - 1),
                sizeof(double) * (num_features + 1) * (num_samples - tree_sample + 1));

        node->left = left;
        node->left_samples = tree_sample - 1;
        node->right = right;
        node->right_samples = num_samples - tree_sample + 1;
        node->feature = tree_feature;
        node->value = tree_value;
    }
}
