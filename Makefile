all:
	gcc -fopenmp -O2 -o maindgem ./dgemv.c -lm
	gcc -fopenmp -O2 -o maininteg ./integ.c -lm
	g++ -fopenmp -o lin ./lin.cpp -lm
dgemv:
	gcc -fopenmp -O2 -o maindgem ./dgemv.c -lm
integ:
	gcc -fopenmp -O2 -o maininteg ./integ.c -lm
lin:
	g++ -fopenmp -o lin ./lin.cpp -lm
