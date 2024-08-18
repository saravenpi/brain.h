#include "brain.h"
#include <pthread.h>
#include <time.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void update_loss(double *loss, brain_t *brain, double *expected)
{
    layer_t *output_layer = brain->layers[brain->nb_layers - 1];

    for (size_t neuron_i = 0; neuron_i < output_layer->nb_neurons;
         neuron_i++) {
        *loss += pow(
            expected[neuron_i] - output_layer->neurons[neuron_i]->output, 2);
    }
}

void display_log(double loss, size_t epoch, time_t start, int epochs)
{
    time_t now = time(NULL);
    int time_elapsed = difftime(now, start);
    int remaining_time = time_elapsed * (epochs - epoch) / epoch;
    int remaining_hours = remaining_time / 3600;
    int remaining_minutes = (remaining_time % 3600) / 60;
    int remaining_seconds = remaining_time % 60;

    printf(
        "Epoch %zu, Time: %ld seconds, Loss: %f, ", epoch, now - start, loss);
    printf("Remaining time: %d hours %d minutes %d seconds\n", remaining_hours,
        remaining_minutes, remaining_seconds);
}

void collect_predictions(
    brain_t *brain, double *predictions, dataset_t training, size_t sample_i)
{
    layer_t *output_layer = brain->layers[brain->nb_layers - 1];

    for (size_t neuron_i = 0; neuron_i < output_layer->nb_neurons;
         neuron_i++)
        predictions[neuron_i] = output_layer->neurons[neuron_i]->output;
}

void *train_thread(void *arg)
{
    thread_data_t *data = (thread_data_t *)arg;
    brain_t *brain = data->brain;
    dataset_t training = data->training;
    int start_sample = data->start_sample;
    int end_sample = data->end_sample;
    double learning_rate = data->learning_rate;
    double epsilon = data->epsilon;
    double local_loss = 0.0;

    for (int i = start_sample; i < end_sample; i++) {
        double *predictions = feedforward(brain, training.samples[i].input);
        double sample_loss = 0.0;
        update_loss(&sample_loss, brain, training.samples[i].output);
        local_loss += sample_loss;
        backpropagate(brain, predictions, training.samples[i].output, epsilon);
        update_weights(brain, learning_rate);
    }
    pthread_mutex_lock(&mutex);
    *data->loss += local_loss;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void train_parallelized(brain_t *brain, dataset_t training, int epochs,
    double learning_rate, double epsilon, int max_cpu_threads)
{
    time_t start;
    time_t end;
    double loss;

    printf("Starting Training...\n");
    start = time(NULL);

    for (int epoch = 0; epoch < epochs; epoch++) {
        loss = 0.0;
        pthread_t threads[max_cpu_threads];
        thread_data_t thread_data[max_cpu_threads];

        int samples_per_thread = training.nb_samples / max_cpu_threads;

        for (int t = 0; t < max_cpu_threads; t++) {
            thread_data[t].brain = brain;
            thread_data[t].training = training;
            thread_data[t].start_sample = t * samples_per_thread;
            thread_data[t].end_sample = (t == max_cpu_threads - 1)
                                            ? training.nb_samples
                                            : (t + 1) * samples_per_thread;
            thread_data[t].learning_rate = learning_rate;
            thread_data[t].epsilon = epsilon;
            thread_data[t].loss = &loss;
            pthread_create(
                &threads[t], NULL, train_thread, (void *)&thread_data[t]);
        }
        for (int t = 0; t < max_cpu_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        if (loss / training.nb_samples < epsilon)
            break;
        display_log(loss / training.nb_samples, epoch, start, epochs);
    }
    end = time(NULL);
    printf(
        "🎉 Finished Training !\n Training time: %ld seconds\nFinal loss: %f \n",
        end - start, loss / training.nb_samples);
}

void train(brain_t *brain, dataset_t training, int epochs,
    double learning_rate, double epsilon)
{
    double *predictions;
    time_t start;
    time_t end;
    double loss;

    printf("✨ Starting Training...\n");
    start = time(NULL);
    for (int epoch = 0; epoch < epochs; epoch++) {
        loss = 0.0;
        for (size_t sample_i = 0; sample_i < training.nb_samples; sample_i++) {
            predictions = feedforward(brain, training.samples[sample_i].input);
            update_loss(&loss, brain, training.samples[sample_i].output);
            backpropagate(brain, predictions,
                training.samples[sample_i].output, epsilon);
            update_weights(brain, learning_rate);
        }
        if (loss / training.nb_samples < epsilon)
            break;
        display_log(loss / training.nb_samples, epoch, start, epochs);
    }
    end = time(NULL);
    printf(
        "Finished Training !\n Training time: %ld seconds\nFinal loss: %f \n",
        end - start, loss / training.nb_samples);
}
