#include "brain.h"

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return x * (1 - x);
}

double relu(double x)
{
    return x < 0 ? 0 : x;
}

double relu_derivative(double x)
{
    return x < 0 ? 0 : 1;
}

double leaky_relu(double x)
{
    return x < 0 ? 0.01 * x : x;
}

double leaky_relu_derivative(double x)
{
    return x < 0 ? 0.01 : 1;
}

double rand_double(void)
{
    srand(time(0));
    return (double)rand() / RAND_MAX;
}
