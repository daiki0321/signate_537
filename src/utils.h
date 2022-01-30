#include <time.h>

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
void* bh_read_file_to_buffer(const char *filename, unsigned int *ret_size);
float sec(clock_t clocks);