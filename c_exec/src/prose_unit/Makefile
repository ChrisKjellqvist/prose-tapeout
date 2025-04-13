CORTEX_M0_DIR=${HOME}/Code/prose_rtl/src/main/resources/AT510-MN-80001-r2p0-00rel0

#### Arm GCC ####
CPU_PRODUCT = CORTEX_M0

# Shared software directory
SOFTWARE_DIR = ${CORTEX_M0_DIR}/software
CMSIS_DIR    = $(SOFTWARE_DIR)/cmsis
CORE_DIR     = $(CMSIS_DIR)/CMSIS/Include

DEVICE_DIR   = $(CMSIS_DIR)/Device/ARM/CMSDK_CM0

# Program file
TESTNAME     ?= prose_e_unit

# Endian Option
COMPILE_BIGEND = 0

# Configuration
USER_DEFINE    = -DCORTEX_M0

DEPS_LIST       = makefile

CPU_TYPE        = -mcpu=cortex-m0

# Startup code directory for gcc
STARTUP_DIR  = $(DEVICE_DIR)/Source/GCC

STARTUP_FILE = startup_CMSDK_CM0
SYSTEM_FILE  = system_CMSDK_CM0

GNG_CC      = arm-none-eabi-g++
GNU_OBJDUMP = arm-none-eabi-objdump
GNU_OBJCOPY = arm-none-eabi-objcopy

BEETHOVEN_SRC = $(BEETHOVEN_ROOT)/Beethoven-Software/src/baremetal.cc \
	       $(BEETHOVEN_ROOT)/Beethoven-Software/src/rocc_cmd.cc \
	       $(BEETHOVEN_ROOT)/Beethoven-Software/src/rocc_response.cc \
	       $(BEETHOVEN_ROOT)/Beethoven-Software/src/util.cc \
	       $(BEETHOVEN_ROOT)/Beethoven-Software/src/alloc.cc \
	       $(BEETHOVEN_ROOT)/Beethoven-Hardware/vsim/generated-src/generated_beethoven_src.cc

LINKER_SCRIPT_PATH = $(SOFTWARE_DIR)/common/scripts
LINKER_SCRIPT = $(LINKER_SCRIPT_PATH)/cmsdk_cm0.ld

GNU_CC_FLAGS = -O3 -fno-exceptions -mthumb $(CPU_TYPE)

.PHONY: all
all:
	$(GNG_CC) $(GNU_CC_FLAGS) $(STARTUP_DIR)/$(STARTUP_FILE).s \
		$(TESTNAME).cc \
		$(BEETHOVEN_SRC) \
		$(SOFTWARE_DIR)/common/retarget/retarget.c \
		$(SOFTWARE_DIR)/common/retarget/uart_stdout.c \
		$(DEVICE_DIR)/Source/$(SYSTEM_FILE).c \
		-I $(DEVICE_DIR)/Include -I $(CORE_DIR) \
		-I $(SOFTWARE_DIR)/common/retarget  \
		-I $(BEETHOVEN_ROOT)/Beethoven-Software/include \
		-I $(BEETHOVEN_ROOT)/Beethoven-Hardware/vsim/generated-src \
		-L $(LINKER_SCRIPT_PATH) \
		-D__STACK_SIZE=0x200 \
		-D__HEAP_SIZE=0x1000 \
		-DBAREMETAL \
		--std=c++17 \
		$(USER_DEFINE) -T $(LINKER_SCRIPT) -o $(TESTNAME).o
	# Generate disassembly code
	$(GNU_OBJDUMP) -S $(TESTNAME).o > $(TESTNAME).lst
	# Generate binary file
	$(GNU_OBJCOPY) -S $(TESTNAME).o -O binary $(TESTNAME).bin
	# Generate hex file
	$(GNU_OBJCOPY) -S $(TESTNAME).o -O verilog $(TESTNAME).hex

clean:
	rm -f programs/*.o
	rm -f programs/*.bin
	rm -f programs/*.lst
	rm -f programs/*.hex
