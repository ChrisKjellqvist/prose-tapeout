OPTS = -I/usr/local/include -std=c++20

all: convert_gptneo gpt_neo/layer_dict.txt
	./convert_gptneo

gpt_neo/layer_dict.txt:
	python3 run_gpt_neo.py

util.o: ../utility/util.cc
	g++ $(OPTS) -c ../utility/util.cc -o util.o

gptneo.o: convert_gptneo.cc
	g++ $(OPTS) -c convert_gptneo.cc -I../utility -o gptneo.o

convert_gptneo: util.o gptneo.o
	g++ $(OPTS) util.o gptneo.o -o convert_gptneo

