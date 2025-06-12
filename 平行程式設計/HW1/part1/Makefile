CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall

all: myexp

logger.o: logger.cpp logger.h PPintrin.h PPintrin.cpp def.h
	$(CXX) $(CXXFLAGS) -c logger.cpp

PPintrin.o: PPintrin.cpp PPintrin.h logger.cpp logger.h def.h
	$(CXX) $(CXXFLAGS) -c PPintrin.cpp

myexp: PPintrin.o logger.o main.cpp serialOP.cpp vectorOP.cpp 
	$(CXX) logger.o PPintrin.o main.cpp serialOP.cpp vectorOP.cpp -o myexp

clean:
	$(RM) *.o *.s myexp *~
