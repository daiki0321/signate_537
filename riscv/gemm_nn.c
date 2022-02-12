#define REG(address) *(volatile unsigned int*)(address)
#define GPIO_BASE  (0xA00C0000)
#define DMEM_BASE_ADDR (0xA0080000)
#define DMEM_LOG_BASE (0xA009F000)

#include <string.h>
// This program is DMEM[0]+DMEM[1]=DMEM[2]
//
//
#define DIM 16
unsigned char* LOG_BASE = (unsigned char*) DMEM_LOG_BASE;
unsigned int* DMEM_BASE = (unsigned int*) DMEM_BASE_ADDR;
unsigned long g_log_writepoint = 0;

void riscv_putc( char c) {
   *(LOG_BASE + g_log_writepoint) = c;
   g_log_writepoint++;
   if (g_log_writepoint >= 4096) {
        g_log_writepoint = 0;
   }
}

void riscv_puts(const char* str) {

   while (*str) {
      riscv_putc(*str++);
   }
}


int main()
{
    int ia, ib, id = 0;

    unsigned int* DMEM_BASE = (unsigned int*) DMEM_BASE_ADDR;
    float (*array_A);
    float (*array_B);
    float (*array_C);
    unsigned int M;
    unsigned int N;
    unsigned int K;
    float ALPHA;
    unsigned int flag;
    float sum;
    volatile int count = 0;
        /* Loop */
    riscv_puts("RiscV is running\n");
    REG(GPIO_BASE) = 0x05;
    while(1) {
         flag = DMEM_BASE[0];
         //flag = (REG(GPIO_BASE) & 0x02);
         count += 1;
         if (flag != 0) {
            REG(GPIO_BASE) = 0x05; // LED1
            riscv_puts("Calc start\n");
            M = DMEM_BASE[1];
            N = DMEM_BASE[2];
            K = DMEM_BASE[3];
            ALPHA = (float) DMEM_BASE[4];
            array_A = (float (*))(DMEM_BASE + 0x10);
            array_B = (float (*))(DMEM_BASE + 0x10 + M * K);
            array_C = (float (*))(DMEM_BASE + 0x10 + M * K + N * K);
            for (ia = 0; ia < M; ia++) {
               for (ib = 0; ib < N; ib++) {
                  sum = 0;
                  for (id = 0; id < K; id++) {
                     sum += array_A[ia * M + id] * array_B[id * K + ib];
                  }
                  array_C[ia * M + ib] = sum;
               }
            }
            riscv_puts("Calc End\n");
            DMEM_BASE[0] = 0;
            REG(GPIO_BASE) = 0x01;
            //DMEM_BASE =  array_C + number * number;
         }
    }
    return 0;
}
