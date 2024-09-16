/*
* Made by saravenpi 2024
* project: brain.h
* file: feedforward.c
*/


#include <stdio.h>

// appends a string to
// the given path,
// it creates the file
// if it doesn't exist
void append_str_to_file(char *str, char *path)
{
    FILE *file;

    file = fopen(path, "a+");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return;
    }
    fprintf(file, "%s", str);
    fclose(file);
}
