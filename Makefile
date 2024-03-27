all: CGaussianPuff

BIN=./bin

# to find location of eigen3 package
# try using command: find /usr -type d -name "eigen3" 
# or: 				 find /opt -type d -name "eigen3"

# EIGEN_DIR=/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 # M1/M2 mac
EIGEN_DIR=/usr/include/eigen3 # For most Windows/Linux systems

# uncomment MAC_FLAG for Mac systems
MAC_FLAG=
# MAC_FLAG=-undefined dynamic_lookup
CGaussianPuff: CGaussianPuff.cpp
	c++ -O3 -I $(EIGEN_DIR) -I /usr/include `python3 -m pybind11 --includes` -shared CGaussianPuff.cpp -o $(BIN)/CGaussianPuff.so -std=c++14 -fPIC $(MAC_FLAG)

clean:
	rm $(BIN)/*.so