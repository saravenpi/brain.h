/*
 * Made by saravenpi 2024
 * project: brain.h
 * file: brain.h
 */

#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef double (*activation_function_t)(double);

struct neuron_s {
	double *w;
	size_t nb_w;
	double *dw;
	double bias;
	double db;
	double output;
	double delta;
	activation_function_t activation;
};
typedef struct neuron_s neuron_t;

struct layer_s {
	struct neuron_s **neurons;
	size_t nb_neurons;
};
typedef struct layer_s layer_t;

struct brain_s {
	struct layer_s **layers;
	size_t nb_layers;
};
typedef struct brain_s brain_t;

struct schema_s {
	size_t *layers;
	size_t nb_layers;
	activation_function_t activation;
};
typedef struct schema_s schema_t;

struct sample_s {
	double *input;
	double *output;
};
typedef struct sample_s sample_t;

struct dataset_s {
	sample_t *samples;
	size_t nb_samples;
};
typedef struct dataset_s dataset_t;

struct thread_data_s {
	brain_t *brain;
	dataset_t training;
	int start_sample;
	int end_sample;
	double learning_rate;
	double epsilon;
	double *loss;
};
typedef struct thread_data_s thread_data_t;

struct training_settings_s {
	double epsilon;
	int epochs;
	int max_cpu_threads;
	double learning_rate;
	dataset_t *dataset;
	char *training_log_file;
};
typedef struct training_settings_s training_settings_t;

double sigmoid(double x);
double sigmoid_derivative(double x);
double rand_double(void);
double relu(double x);
double relu_derivative(double x);
double leaky_relu(double x);
double leaky_relu_derivative(double x);

brain_t *create_brain(schema_t schema);
void initialize_weights(brain_t *brain);
void free_brain(brain_t *brain);

double *feedforward(brain_t *brain, double *input);
void update_weights(brain_t *brain, double learning_rate);
void backpropagate(
	brain_t *brain, double *predictions, double *expected, double epsilon);

void train(brain_t *brain, training_settings_t settings);
void train_parallelized(brain_t *brain, training_settings_t settings);

void save_brain(brain_t *brain, const char *filename);
brain_t *load_brain(const char *filename, schema_t schema);
