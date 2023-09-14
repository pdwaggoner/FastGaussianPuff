all: CGaussianPuff

CGaussianPuff: CGaussianPuff.cpp
	c++ -O3 -I /usr/include/eigen3 -shared CGaussianPuff.cpp -o CGaussianPuff.so -std=c++14 -fPIC -fopenmp -I/usr/include `python3 -m pybind11 --includes`
clean:
	rm *.so