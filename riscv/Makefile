
CFLAGS := -O3 -Wno-implicit
CFLAGS += -fno-builtin-printf -fno-common

SRC = gemm_nn.c start.s
HDR = 
LDFLAGS := -std=gnu99 -static -lgcc -lc -Tlinker.ld -march=rv32imf -mabi=ilp32f -Wa,-march=rv32imf -nostartfiles

main: $(SRC) $(HDR)
	riscv64-unknown-elf-gcc $(CFLAGS) $(SRC) $(LDFLAGS) -o main
	riscv64-unknown-elf-objcopy --gap-fill 0 -O verilog main main.hex
	riscv64-unknown-elf-objcopy  -O binary main main.bin
	riscv64-unknown-elf-objdump -D main > main.dump
	/usr/bin/python3 ./hex_dump.py main


clean:
	rm -f main main.hex main.dump

