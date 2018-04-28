typedef struct _node_t {
    struct _node_t* left;
    struct _node_t* right;
    unsigned long num_samples;
    unsigned long num_features;
    double** X;
    double value;
    unsigned long feature;
} node_t;

void train(node_t* root, unsigned long max_depth, unsigned long min_size,
        double lambda, double gamma, double learning_rate);

double* predict(unsigned long num_samples, double** X, node_t* model, bool
        class_flag);
