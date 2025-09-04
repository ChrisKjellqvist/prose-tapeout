# super annoying torch includes... you'll probably have to change for your system
TORCH_OPTS = -I/usr/local/include/torch/csrc/api/include/ -I/usr/local/include/ 
OPTS = $(TORCH_OPTS) -O2 -std=c++20
LINK_OPTS = -ltorch_cpu -ltorch -ltorch_global_deps -lc10
ifeq ($(shell uname), Darwin)
	LINK_OPTS += -rpath /usr/local/lib
endif

.PHONY: all
all: convert_gptneo gpt_neo/layer_dict.txt
	./convert_gptneo
	cp prose_rptr.h ../deploy/include

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