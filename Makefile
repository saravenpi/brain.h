#
# Made by saravenpi 2024
# project: brain.h
# file: brain.h
#

CC = gcc
CFLAGS = -Wall -Werror -I.

LIB_SRCS := src/backpropagate.c \
			src/brain.c \
			src/feedforward.c \
			src/file.c \
			src/train.c \
			src/utils.c

LIB_OBJS = $(LIB_SRCS:.c=.o)
LIB_NAME = libbrain.so

all: $(LIB_NAME)

$(LIB_NAME): $(LIB_OBJS)
	$(CC) -shared -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

clean:
	rm -f $(LIB_OBJS)

fclean: clean
	rm -f $(LIB_NAME)

re: fclean all
