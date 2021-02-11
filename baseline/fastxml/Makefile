INC=-I../tools/c++
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread

all: clean fastxml_train fastxml_predict

fastXML_train:
	g++ -o fastxml_train $(CXXFLAGS) $(INC) fastxml_train.cpp fastxml.cpp ../tools/c++/mat.cpp $(LIBFLAGS)

fastXML_predict:
	g++ -o fastxml_predict $(CXXFLAGS) $(INC) fastxml_predict.cpp fastxml.cpp ../tools/c++/mat.cpp $(LIBFLAGS)

clean:
	rm -f fastxml_train fastxml_predict

