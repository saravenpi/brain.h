#include "brain.h"

void update_neuron(layer_t *prev_layer, neuron_t *neuron)
{
    double sum = 0.0;

    for (size_t i = 0; i < prev_layer->nb_neurons; i++)
        sum += prev_layer->neurons[i]->output * neuron->w[i];
    sum += neuron->bias;
    neuron->output = neuron->activation(sum);
}

void update_layer(layer_t *layer, layer_t *prev_layer)
{
    for (size_t i = 0; i < layer->nb_neurons; i++)
        update_neuron(prev_layer, layer->neurons[i]);
}

double *feedforward(brain_t *brain, double *input)
{
    layer_t *current_layer;
    layer_t *prev_layer;
    size_t output_size = brain->layers[brain->nb_layers - 1]->nb_neurons;
    double *output = malloc(output_size * sizeof(double));

    if (!output) {
        perror("Failed to allocate memory for output");
        exit(EXIT_FAILURE);
    }
    layer_t *input_layer = brain->layers[0];
    for (size_t i = 0; i < input_layer->nb_neurons; i++)
        input_layer->neurons[i]->output = input[i];
    for (size_t i = 1; i < brain->nb_layers; i++) {
        current_layer = brain->layers[i];
        prev_layer = brain->layers[i - 1];
        update_layer(current_layer, prev_layer);
    }
    layer_t *output_layer = brain->layers[brain->nb_layers - 1];
    for (size_t i = 0; i < output_size; i++)
        output[i] = output_layer->neurons[i]->output;
    return output;
}
