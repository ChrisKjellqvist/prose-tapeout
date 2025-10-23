# super annoying torch includes... you'll probably have to change for your system
TORCH_OPTS = -I/usr/local/include/torch/csrc/api/include/ -I/usr/local/include/ 
OPTS = $(TORCH_OPTS) -O0 -g -std=c++20 -DUSE_TORCH
LINK_OPTS = -ltorch_cpu -ltorch -ltorch_global_deps -lc10
ifeq ($(shell uname), Darwin)
	LINK_OPTS += -rpath /usr/local/lib
endif

.PHONY: all
all: convert_gptneo gpt_neo/layer_dict.txt
	LD_LIBRARY_PATH=/usr/local/lib;./convert_gptneo
	cp prose_rptr.h ../deploy/include
	cp prose_rptr.cc ../deploy/lib
	xxd -c 64 -l 16384 ../model/gpt_neo/prose_input.raw
	@echo "You should see some debugy-looking outputs above (all-zeroes is wrong)"

gpt_neo/layer_dict.txt:
	python run_gpt_neo.py

util.o: ../utility/util.cc
	g++ $(OPTS) -c ../utility/util.cc -o util.o

gptneo.o: convert_gptneo.cc
	g++ $(OPTS) -c convert_gptneo.cc -I../utility -o gptneo.o

convert_gptneo: util.o gptneo.o
	g++ $(OPTS) $(LINK_OPTS) util.o gptneo.o -o convert_gptneo

.PHONY: clean
clean:
	rm -f util.o gptneo.o convert_gptneo gpt_neo/layer_dict.txt
