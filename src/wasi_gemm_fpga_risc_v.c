#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>

#define REG(address) *(volatile unsigned int*)(address)


// GPIO[0]=RISC_V_IMEM_RESET RISC_V_DMEM_RESET
// GPIO[1]=LED0
// GPIO[2]=LED1
// This program is DMEM[0]+DMEM[1]=DMEM[2]



x_gemm_fpga_risc_v(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    unsigned int c;
    //DMEM
    int uio0_fd = open("/dev/uio0", O_RDWR | O_SYNC);
    unsigned int* DMEM_BASE = (unsigned int*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);
    unsigned char* UDMEM_BASE = (unsigned char*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);

    //DMEM1
    int uio1_fd = open("/dev/uio1", O_RDWR | O_SYNC);
    unsigned int* DMEM1_BASE = (unsigned int*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio1_fd, 0);
    unsigned char* UDMEM1_BASE = (unsigned char*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio1_fd, 0);

    //IMEM
    int uio2_fd = open("/dev/uio2", O_RDWR | O_SYNC);
    unsigned int* IMEM_BASE = (unsigned int*) mmap(NULL, 0x40000, PROT_READ|PROT_WRITE, MAP_SHARED, uio2_fd, 0);

    //IMEM1
    int uio3_fd = open("/dev/uio3", O_RDWR | O_SYNC);
    unsigned int* IMEM1_BASE = (unsigned int*) mmap(NULL, 0x40000, PROT_READ|PROT_WRITE, MAP_SHARED, uio3_fd, 0);

    //GPIO
    int uio4_fd = open("/dev/uio4", O_RDWR | O_SYNC);
    unsigned int* GPIO_DATA = (unsigned int*) mmap(NULL, 0x4000, PROT_READ|PROT_WRITE, MAP_SHARED, uio4_fd, 0);
    unsigned int* GPIO_TRI = GPIO_DATA + 1;


    FILE* fd = fopen("main", "rb");
    if(fd == NULL) {
	  printf("main file not opened\n");
	  return -1;
    }
    if (fseek(fd, 0x1000 , SEEK_END) != 0) {
        return -1;

    } 
    int file_size = ftell(fd);
    if (file_size == -1) {
        return -1;
    }
   
    printf("main File size is %d\n", file_size);
    fseek(fd, 0x1000, SEEK_SET);
    
    FILE* fd1 = fopen("nuttx_1.bin", "rb");
    if(fd1 == NULL) {
	printf("file not opened\n");
	return -1;	
    }
	        
    if (fseek(fd1, 0 , SEEK_END) != 0) {
	return -1;
    }
		   
    int file_size1 = ftell(fd1);
    if (file_size1 == -1) {
	return -1;
    }
    printf("nuttx_1 File size is %d\n", file_size1);
    fseek(fd1, 0x0, SEEK_SET);

    FILE* fd2 = fopen("test.wasm", "rb");
    if(fd2 == NULL) {
	printf("file not opened\n");
	return -1;
    }

    if (fseek(fd2, 0 , SEEK_END) != 0) {
	    return -1;
    }


    int file_size2 = ftell(fd2);
    if (file_size2 == -1) {
	    return -1;
    }
    printf("wasm File size wa:s %d\n", file_size2);

    fseek(fd2, 0x0, SEEK_SET);

    REG(GPIO_TRI) = 0x00;
    REG(GPIO_DATA) = 0x02; // LED0

    /* Memory access test */
    
    c = 0; 
   
    //DMEM_BASE[0] = 1; // start flag
    DMEM_BASE[1] = 3; // matrix number
    unsigned int (*MatrixA)[3] = (unsigned int (*)[3]) &DMEM_BASE[2];
    unsigned int (*MatrixB)[3] = (unsigned int (*)[3]) &DMEM_BASE[2 + 3*3];

    for (int i = 0; i < 3; i++ ) {
       for (int j = 0; j < 3; j++) { 
          MatrixA[i][j] = 1 + i * 3 + j;
          MatrixB[i][j] = 9 - (i * 3 + j);
       }
    }

    
    
    DMEM1_BASE[0] = 0x00000056;
    DMEM1_BASE[1] = 0x00000078;
    DMEM1_BASE[2] = 0x00000000;
    
    int rc = 0;  
    rc = fread(IMEM_BASE, sizeof(unsigned int), 0x40000 / sizeof(unsigned int), fd);
    
    if (rc < 0) {
	    printf("Main File Read IMEM error\n");
	    return -1;
    }
    
    rc = fread(IMEM1_BASE, sizeof(unsigned int), 0x40000 / sizeof(unsigned int), fd1);


    if (rc < 0) {
         printf("File Read IMEM error\n");
	 return -1;
    } 

    rc = fread(UDMEM1_BASE+0x4000, sizeof(unsigned int), 0x4000 / sizeof(unsigned int), fd2);
        if (rc < 0) {
		         printf("File Read IMEM error\n");
			          return -1;
				      }

    
    sleep(1);
    unsigned int i1 = IMEM_BASE[0];
    unsigned int i2 = IMEM_BASE[1];
    unsigned int i3 = IMEM_BASE[2];
    printf("%x %x %x\n", i1,i2,i3);

    unsigned int ii1 = IMEM1_BASE[0];
    unsigned int ii2 = IMEM1_BASE[1];
    unsigned int ii3 = IMEM1_BASE[2];
    printf("%x %x %x\n", ii1,ii2,ii3);

    unsigned int d1 = UDMEM1_BASE[0x4000];
    unsigned int d2 = UDMEM1_BASE[0x4001];
    unsigned int d3 = UDMEM1_BASE[0x4002];
    printf("%x %x %x\n", d1,d2,d3);
    DMEM_BASE[0] = 1; // start flag

    
    sleep(1);
      REG(GPIO_DATA) = 0x04; // LED1
    sleep(1);
      REG(GPIO_DATA) = 0x01; // Reset off
//    sleep(4);
//    REG(GPIO_DATA) = 0x00; // Reset on
   
    sleep(2); 
    unsigned int c1 = DMEM_BASE[0];
    unsigned int c2 = DMEM_BASE[1];

    printf("%x %x \n\r",c1, c2);

    for (int i = 0; i < 3*3*3; i++) {
       printf("%x ", DMEM_BASE[2 + i]);
    }
    printf("\n");

    
    for (int i = 0; i < 3; i++ ) {
      for (int j = 0; j < 3; j++) {
        MatrixA[i][j] = 2;
        MatrixB[i][j] = 1;
	}
    }
    DMEM_BASE[0] = 1;    

    c1 = DMEM_BASE[0];
    c2 = DMEM_BASE[1];

    printf("%x %x \n\r",c1, c2);

    for (int i = 0; i < 3*3*3; i++) {
        printf("%x ", DMEM_BASE[2 + i]);
    }
    printf("\n");


    unsigned int ci1 = DMEM1_BASE[0];
    unsigned int ci2 = DMEM1_BASE[1];
    unsigned int ci3 = DMEM1_BASE[2];
    printf("%x %x %x\n\r",ci1, ci2, ci3);    
    
    for ( int i = 0; i < 5000; i++) {
	    unsigned char progress = UDMEM_BASE[0x10000 + i];
	    printf("%c", progress);
    }
    printf("\n");

    for ( int i = 0; i < 5000; i++) {
	    unsigned char progress1 = UDMEM1_BASE[0xC + i];
	    printf("%c", progress1);
    }
    printf("\n");


    sleep(1);
    fclose(fd);
    fclose(fd1);
    fclose(fd2);
    return 0;
}

