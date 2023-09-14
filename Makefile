all: CGaussianPuff

BIN=./bin
EIGEN_DIR=/usr/include/eigen3

CGaussianPuff: CGaussianPuff.cpp
	c++ -O3 -I $(EIGEN_DIR) -I /usr/include `python3 -m pybind11 --includes` -shared CGaussianPuff.cpp -o $(BIN)/CGaussianPuff.so -std=c++14 -fPIC
clean:
	rm $(BIN)/*.so