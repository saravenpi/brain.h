CC = gcc
CFLAGS = -Wall -Werror -I.

SRC_DIR = src
LIB_SRCS := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*/*.c)

LIB_OBJS = $(LIB_SRCS:.c=.o)
LIB_NAME = libbrain.so

all: $(LIB_NAME)

$(LIB_NAME): $(LIB_OBJS)
	$(CC) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

clean:
	rm -f $(LIB_OBJS)

fclean: clean
	rm -f $(LIB_NAME)

re: fclean all
