/*
* Made by saravenpi 2024
* project: brain.h
* file: train.c
*/

#include "brain.h"
#include <pthread.h>
#include <time.h>

void append_str_to_file(char *str, char *path);

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

void display_log(double loss, size_t epoch, time_t start, int epochs,
    char *training_log_file)
{
    time_t now = time(NULL);
    int time_elapsed = difftime(now, start);
    int remaining_time = time_elapsed * (epochs - epoch) / epoch;

    int remaining_hours = remaining_time / 3600;
    int remaining_minutes = (remaining_time % 3600) / 60;
    int remaining_seconds = remaining_time % 60;
    char log_line[512];

    sprintf(log_line, "%zu %f\n", epoch, loss);
    printf(
        "Epoch %zu, Time: %ld seconds, Loss: %f, ", epoch, now - start, loss);
    printf("Remaining time: %d hours %d minutes %d seconds\n", remaining_hours,
        remaining_minutes, remaining_seconds);
    if (training_log_file != NULL) {
        append_str_to_file(log_line, training_log_file);
    }
}

void collect_predictions(
    brain_t *brain, double *predictions, dataset_t training, size_t sample_i)
{
    layer_t *output_layer = brain->layers[brain->nb_layers - 1];

    for (size_t neuron_i = 0; neuron_i < output_layer->nb_neurons; neuron_i++)
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
    pthread_t thread_id = pthread_self();

    for (int i = start_sample; i < end_sample; i++) {
        double *predictions = feedforward(brain, training.samples[i].input);
        double sample_loss = 0.0;
        update_loss(&sample_loss, brain, training.samples[i].output);
        local_loss += sample_loss;
        backpropagate(brain, predictions, training.samples[i].output, epsilon);
        update_weights(brain, learning_rate);
    }
    printf("[Thread %lu] Finished Training Loop\n", thread_id);
    pthread_mutex_lock(&mutex);
    *data->loss += local_loss;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

void train_parallelized(brain_t *brain, training_settings_t settings)
{
    time_t start;
    time_t end;
    double loss;

    printf("✨ Starting Parallelized Training...\n");
    start = time(NULL);
    for (int epoch = 0; epoch < settings.epochs; epoch++) {
        loss = 0.0;
        pthread_t threads[settings.max_cpu_threads];
        thread_data_t thread_data[settings.max_cpu_threads];

        int samples_per_thread =
            settings.dataset->nb_samples / settings.max_cpu_threads;

        for (int t = 0; t < settings.max_cpu_threads; t++) {
            thread_data[t].brain = brain;
            thread_data[t].training = *settings.dataset;
            thread_data[t].start_sample = t * samples_per_thread;
            thread_data[t].end_sample = (t == settings.max_cpu_threads - 1)
                                            ? settings.dataset->nb_samples
                                            : (t + 1) * samples_per_thread;
            thread_data[t].learning_rate = settings.learning_rate;
            thread_data[t].epsilon = settings.epsilon;
            thread_data[t].loss = &loss;
            pthread_create(
                &threads[t], NULL, train_thread, (void *)&thread_data[t]);
        }
        for (int t = 0; t < settings.max_cpu_threads; t++) {
            pthread_join(threads[t], NULL);
        }
        if (loss / settings.dataset->nb_samples < settings.epsilon)
            break;
        display_log(loss / settings.dataset->nb_samples, epoch + 1, start,
            settings.epochs, settings.training_log_file);
    }
    end = time(NULL);
    printf("🎉 Finished Training !\n Training time: %ld seconds\nFinal loss: "
           "%f \n",
        end - start, loss / settings.dataset->nb_samples);
}

void train(brain_t *brain, training_settings_t settings)
{
    double *predictions;
    time_t start;
    time_t end;
    double loss;

    printf("✨ Starting Training...\n");
    start = time(NULL);
    for (int epoch = 0; epoch < settings.epochs; epoch++) {
        loss = 0.0;
        for (size_t sample_i = 0; sample_i < settings.dataset->nb_samples;
             sample_i++) {
            predictions =
                feedforward(brain, settings.dataset->samples[sample_i].input);
            update_loss(
                &loss, brain, settings.dataset->samples[sample_i].output);
            backpropagate(brain, predictions,
                settings.dataset->samples[sample_i].output, settings.epsilon);
            update_weights(brain, settings.learning_rate);
        }
        if (loss / settings.dataset->nb_samples < settings.epsilon)
            break;
        display_log(loss / settings.dataset->nb_samples, epoch + 1, start,
            settings.epochs, settings.training_log_file);
    }
    end = time(NULL);
    printf("🎉 Finished Training !\n Training time: %ld seconds\nFinal loss: "
           "%f \n",
        end - start, loss / settings.dataset->nb_samples);
    printf("📄 Training logs are saved in %s\n", settings.training_log_file);
}
