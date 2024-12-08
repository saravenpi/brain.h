/*
* Made by saravenpi 2024
* project: brain.h
* file: backpropagate.c
*/

#include "brain.h"

void update_weights(brain_t *brain, double learning_rate)
{
    layer_t *layer;
    neuron_t *neuron;

    for (size_t layer_i = 1; layer_i < brain->nb_layers; layer_i++) {
        layer = brain->layers[layer_i];
        for (size_t neuron_i = 0; neuron_i < layer->nb_neurons; neuron_i++) {
            neuron = layer->neurons[neuron_i];
            for (size_t weight_i = 0; weight_i < neuron->nb_w; weight_i++)
                neuron->w[weight_i] += learning_rate * neuron->dw[weight_i];
            neuron->bias += learning_rate * neuron->db;
        }
    }
}

void calculate_output_layer_error_and_delta(
    brain_t *brain, double *predictions, double *expected)
{
    layer_t *output_layer = brain->layers[brain->nb_layers - 1];
    neuron_t *neuron;
    double error;

    for (size_t neuron_i = 0; neuron_i < output_layer->nb_neurons;
         neuron_i++) {
        neuron = output_layer->neurons[neuron_i];
        error = expected[neuron_i] - predictions[neuron_i];
        if (neuron->activation == sigmoid)
            neuron->delta = error * sigmoid_derivative(neuron->output);
        else if (neuron->activation == relu)
            neuron->delta = error * relu_derivative(neuron->output);
        else if (neuron->activation == leaky_relu)
            neuron->delta = error * leaky_relu_derivative(neuron->output);
    }
}

void calculate_hidden_errors_and_deltas(
    brain_t *brain, double *predictions, double *expected)
{
    layer_t *layer;
    layer_t *next_layer;
    neuron_t *neuron;
    neuron_t *next_neuron;
    double error;

    for (size_t layer_i = brain->nb_layers - 2; layer_i > 0; layer_i--) {
        layer = brain->layers[layer_i];
        next_layer = brain->layers[layer_i + 1];
        for (size_t neuron_i = 0; neuron_i < layer->nb_neurons; neuron_i++) {
            neuron = layer->neurons[neuron_i];
            error = 0.0;
            for (size_t next_neuron_i = 0;
                 next_neuron_i < next_layer->nb_neurons; next_neuron_i++) {
                next_neuron = next_layer->neurons[next_neuron_i];
                error += next_neuron->delta * next_neuron->w[neuron_i];
            }
            if (neuron->activation == sigmoid)
                neuron->delta = error * sigmoid_derivative(neuron->output);
            else if (neuron->activation == relu)
                neuron->delta = error * relu_derivative(neuron->output);
            else if (neuron->activation == leaky_relu)
                neuron->delta = error * leaky_relu_derivative(neuron->output);
        }
    }
}

void calculate_gradients_weights_and_biases(brain_t *brain, double epsilon)
{
    layer_t *layer;
    layer_t *prev_layer;
    neuron_t *neuron;

    for (size_t layer_i = 1; layer_i < brain->nb_layers; layer_i++) {
        layer = brain->layers[layer_i];
        prev_layer = brain->layers[layer_i - 1];
        for (size_t neuron_i = 0; neuron_i < layer->nb_neurons; neuron_i++) {
            neuron = layer->neurons[neuron_i];
            for (size_t weight_i = 0; weight_i < neuron->nb_w; weight_i++) {
                neuron->dw[weight_i] =
                    neuron->delta *
                    (prev_layer->neurons[weight_i]->output + epsilon);
            }
            neuron->db = neuron->delta;
        }
    }
}

void backpropagate(
    brain_t *brain, double *predictions, double *expected, double epsilon)
{
    calculate_output_layer_error_and_delta(brain, predictions, expected);
    calculate_hidden_errors_and_deltas(brain, predictions, expected);
    calculate_gradients_weights_and_biases(brain, epsilon);
}
