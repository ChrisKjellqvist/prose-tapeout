all: convert_gptneo gpt_neo/layer_dict.txt
	./convert_gptneo

gpt_neo/layer_dict.txt:
	python3 run_gpt_neo.py

util.o: ../utility/util.cc
	g++ -std=c++20 -c ../utility/util.cc -o util.o

gptneo.o: convert_gptneo.cc
	g++ -std=c++20 -c convert_gptneo.cc -I../utility -o gptneo.o

convert_gptneo: util.o gptneo.o
	g++ -std=c++20 util.o gptneo.o -o convert_gptneo

