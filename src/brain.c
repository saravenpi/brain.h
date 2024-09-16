/*
* Made by saravenpi 2024
* project: brain.h
* file: brain.c
*/

#include "brain.h"

neuron_t *create_neuron(size_t nb_inputs, activation_function_t activation)
{
    neuron_t *neuron = (neuron_t *)malloc(sizeof(neuron_t));

    if (!neuron) {
        perror("Failed to allocate memory for neuron");
        exit(EXIT_FAILURE);
    }
    neuron->w = (double *)malloc(nb_inputs * sizeof(double));
    if (!neuron->w) {
        free(neuron);
        perror("Failed to allocate memory for neuron");
        exit(EXIT_FAILURE);
    }
    neuron->dw = (double *)malloc(nb_inputs * sizeof(double));
    if (!neuron->dw) {
        free(neuron->w);
        free(neuron);
        perror("Failed to allocate memory for neuron");
        exit(EXIT_FAILURE);
    }
    neuron->nb_w = nb_inputs;
    neuron->bias = 0.0;
    neuron->db = 0.0;
    neuron->output = 0.0;
    neuron->delta = 0.0;
    neuron->activation = activation;
    return neuron;
}

layer_t *create_layer(size_t nb_neurons, size_t nb_inputs_per_neuron,
    activation_function_t activation)
{
    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));

    if (!layer) {
        perror("Failed to allocate memory for layer");
        exit(EXIT_FAILURE);
    }
    layer->neurons = (neuron_t **)malloc(nb_neurons * sizeof(neuron_t *));
    if (!layer->neurons) {
        free(layer);
        perror("Failed to allocate memory for neurons array");
        exit(EXIT_FAILURE);
    }
    layer->nb_neurons = nb_neurons;
    for (size_t i = 0; i < nb_neurons; i++)
        layer->neurons[i] = create_neuron(nb_inputs_per_neuron, activation);
    return layer;
}

brain_t *create_brain(schema_t schema)
{
    brain_t *brain = (brain_t *)malloc(sizeof(brain_t));
    size_t nb_inputs;

    if (!brain) {
        perror("Failed to allocate memory for brain");
        exit(EXIT_FAILURE);
    }
    printf("🧠 Creating brain with %zu layers\n", schema.nb_layers);
    brain->layers = (layer_t **)malloc(schema.nb_layers * sizeof(layer_t *));
    if (!brain->layers) {
        free(brain);
        perror("Failed to allocate memory for layers array");
        exit(EXIT_FAILURE);
    }
    brain->nb_layers = schema.nb_layers;
    for (size_t i = 0; i < schema.nb_layers; i++) {
        nb_inputs = (i == 0) ? 0 : schema.layers[i - 1];
        brain->layers[i] = create_layer(schema.layers[i], nb_inputs, sigmoid);
    }
    printf("🧠 Brain created\n");
    return brain;
}

void initialize_weights(brain_t *brain)
{
    layer_t *layer;
    neuron_t *neuron;

    for (size_t i = 1; i < brain->nb_layers; i++) {
        layer = brain->layers[i];
        for (size_t j = 0; j < layer->nb_neurons; j++) {
            neuron = layer->neurons[j];
            for (size_t k = 0; k < neuron->nb_w; k++) {
                neuron->w[k] = ((double)rand() / RAND_MAX) * 2 - 1;
                neuron->dw[k] = 0.0;
            }
            neuron->bias = ((double)rand() / RAND_MAX) * 2 - 1;
            neuron->db = 0.0;
        }
    }
}

void free_brain(brain_t *brain)
{
    layer_t *layer;
    neuron_t *neuron;

    for (size_t i = 0; i < brain->nb_layers; i++) {
        layer = brain->layers[i];
        for (size_t j = 0; j < layer->nb_neurons; j++) {
            neuron = layer->neurons[j];
            free(neuron->w);
            free(neuron->dw);
            free(neuron);
        }
        free(layer->neurons);
        free(layer);
    }
    free(brain->layers);
    free(brain);
}

void save_brain(brain_t *brain, const char *filename)
{
    FILE *file = fopen(filename, "wb");
    layer_t *layer;
    neuron_t *neuron;

    if (!file) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < brain->nb_layers; i++) {
        layer = brain->layers[i];
        for (size_t j = 0; j < layer->nb_neurons; j++) {
            neuron = layer->neurons[j];
            fwrite(neuron->w, sizeof(double), neuron->nb_w, file);
            fwrite(&neuron->bias, sizeof(double), 1, file);
        }
    }
    fclose(file);
    printf("💾 Model saved to %s\n", filename);
}

brain_t *load_brain(const char *filename, schema_t schema)
{
    FILE *file = fopen(filename, "rb");
    brain_t *brain;
    layer_t *layer;
    neuron_t *neuron;

    if (!file) {
        perror("Failed to open model file");
        exit(EXIT_FAILURE);
    }
    brain = create_brain(schema);
    for (size_t i = 0; i < brain->nb_layers; i++) {
        layer = brain->layers[i];
        for (size_t j = 0; j < layer->nb_neurons; j++) {
            neuron = layer->neurons[j];
            fread(neuron->w, sizeof(double), neuron->nb_w, file);
            fread(&neuron->bias, sizeof(double), 1, file);
        }
    }
    fclose(file);
    printf("💾 Model loaded from %s\n", filename);
    return brain;
}
