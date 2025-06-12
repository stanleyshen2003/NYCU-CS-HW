default: conv

CC = gcc
FLAGS = -O3 -DCL_TARGET_OPENCL_VERSION=220 -lOpenCL -m64 -ffloat-store -w -g

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -o $@ $(OBJS) $(FLAGS)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

clean:
	$(RM) conv *.o output.bmp ref.bmp
