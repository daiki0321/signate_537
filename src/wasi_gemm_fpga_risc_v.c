#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <string.h>
#include <fcntl.h>

#define REG(address) *(volatile unsigned int*)(address)


// GPIO[0]=RISC_V_IMEM_RESET RISC_V_DMEM_RESET
// GPIO[1]=LED0
// GPIO[2]=LED1
// This program is DMEM[0]+DMEM[1]=DMEM[2]
#define DIM  16

static int uio4_fd = 0;
volatile unsigned int* GPIO_BASE;

static void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc,
        volatile unsigned int* DMEM_BASE)
{

    //DMEM_BASE[0] = 1; // start flag
    volatile int i = 0;
    int calctime = 0;
    volatile float *aa = (volatile float*)(DMEM_BASE + 16);
    volatile unsigned char *a = (volatile unsigned char*)(DMEM_BASE + 16);
    volatile float *bb = (volatile float*)(DMEM_BASE + 16 + DIM * DIM);
    volatile float *cc = (volatile float*)(DMEM_BASE + 16 + DIM * DIM + DIM * DIM) ;
    int mm, kk, nn;
    volatile unsigned char * UDMEM_BASE = (volatile unsigned char *) DMEM_BASE;
    

    if (uio4_fd == 0) {
    uio4_fd = open("/dev/uio4", O_RDWR | O_SYNC);
    GPIO_BASE = (unsigned int*) mmap(NULL, 0x2000, PROT_READ|PROT_WRITE, MAP_SHARED, uio4_fd, 0);
    }
    printf("Start\n");

    i= 0;
    for (int ii = 0; ii < M; ii+= DIM) {
        mm = ((ii + DIM) > M ? (M - ii): DIM);
        for (int tt = 0; tt < K; tt+= DIM) {
            kk = ((tt + DIM) > K ? (K - tt): DIM);     
            if(kk != DIM) {
                for (int i_ = 0; i_ < mm; i_ ++) {
                    for (int j_ = kk; j_ < DIM; j_ ++) {
                        aa[i_ * DIM + j_] = 0;
                    }
                }                
            }
            for (int i_ = 0; i_ < mm; i_ ++) {
                for (int j_ = 0; j_ < kk; j_ ++) {
                    aa[i_ * DIM + j_] = A[(ii + i_) * K + (tt + j_)];
                }
            }
            for (int jj = 0; jj < N; jj += DIM) {
                nn = ((jj + DIM) > N ? (N - jj): DIM);
                for (int i_ = 0; i_ < kk; i_ ++) {
                    for (int j_ = 0; j_ < nn; j_ ++) {
                        bb[i_ * DIM + j_] = B[(tt + i_) * N + (jj + j_)];
                    }
                }
                i++;              
                DMEM_BASE[1] = DIM; //mm;
                DMEM_BASE[2] = DIM; //nn;
                DMEM_BASE[3] = DIM; //kk;
                DMEM_BASE[4] = ALPHA;
                DMEM_BASE[0] = 1; // start flag
                REG(GPIO_BASE) = 0x03;
                volatile int count = 0;
                while (DMEM_BASE[0]) {
                    usleep(1000);
                    count++;
                }
#ifdef DEBUG_CALC
                printf("%dms (%d/%d)\r", count, i, M * N * K/(DIM * DIM * DIM));
#endif                
                for (int i_ = 0; i_ < mm; i_ ++) {
                    for (int j_ = 0; j_ < nn; j_ ++) {
                        C[(ii + i_) * N + (jj +j_)] += cc [i_ * DIM + j_];
                    }
                }
            }
        }
    }
}

static void gemm_nn_cpu(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

static int uio0_fd = 0;
static int uio1_fd = 0;
volatile unsigned int* DMEM_BASE;
volatile unsigned int* DMEM1_BASE;

x_gemm_fpga_risc_v(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{


     //DMEM
     if (uio0_fd == 0) {
         printf("Open DMEM\n");
        uio0_fd = open("/dev/uio0", O_RDWR | O_SYNC);
        DMEM_BASE = (unsigned int*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);
     }


    //DMEM1
    if (uio1_fd == 0) {
        printf("Open DMEM1\n");
        uio1_fd = open("/dev/uio1", O_RDWR | O_SYNC);
        DMEM1_BASE = (unsigned int*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio1_fd, 0);
    } 

    printf("riscv: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);


    if(!TA && !TB) {
#ifdef DEBUG_CALC        
        float * testC;

        testC = (float*) malloc( M * N * 4);
        if(testC == NULL) {
            printf("malloc failed\n");
        }
        for(int i = 0; i < M; ++i){
            for(int j = 0; j < N; ++j){
                testC[i*ldc + j] = C[i*ldc + j];
            }
        }  
#endif              
        gemm_nn(M, N, K, ALPHA, A ,lda, B, ldb, C, ldc, DMEM_BASE);
#ifdef DEBUG_CALC        
        printf ("riscv finished\n");
        
        gemm_nn_cpu(M, N, K, ALPHA, A ,lda, B, ldb, testC, ldc);
        printf ("cpu  finished\n");
        for(unsigned int i = 0; i < M; ++i) {
            for(unsigned int j = 0; j < N; ++j){
                if ((testC[i*ldc + j] - C[i*ldc + j]) * (testC[i*ldc + j] - C[i*ldc + j]) > 0.00001) {    
                    printf ("Calclation wrong CPU = %f, RISCV = %f, I = %d, J = %d\n", testC[i*ldc + j], C[i*ldc + j], i, j);
                }
            }
        }
        free (testC);
        printf ("Comparison Finished\n"); 
#endif        
    }
    else if(TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, DMEM_BASE);
    else if(!TA && TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, DMEM_BASE);
    else
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, DMEM_BASE);


}
