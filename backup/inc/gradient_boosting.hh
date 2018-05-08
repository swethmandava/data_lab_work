#ifndef __GRADIENT_BOOSTING_HH__
#define __GRADIENT_BOOSTING_HH__

#include "tree.hh"
#include <fstream>

node_t** gradient_boosting(char* filename, unsigned long num_samples,
                       unsigned long epochs, double learning_rate,
                       unsigned long max_depth, unsigned long min_size,
                       double lambda, double gamma,
                       unsigned long batch_size, unsigned long num_features,
                       unsigned long cols);

void readcsv(std::ifstream& file, unsigned long index, node_t* root, unsigned
        long cols);

double* predict_gboost(node_t* root, node_t** weak_learners, unsigned long
        num, unsigned long cols);


#endif /* end of include guard: __GRADIENT_BOOSTING_HH__ */
