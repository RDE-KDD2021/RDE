SRC=../fastxml
CXXFLAGS=-std=c++11 -O3 -g
LIBFLAGS=-pthread
CXXINCPATH=../tools/c++
INC=-I$(CXXINCPATH) -I$(SRC)

all: clean pfastrexml_train pfastrexml_predict

PfastreXML_train:
	$(CXX) -o pfastrexml_train $(CXXFLAGS) $(INC) pfastrexml_train.cpp $(SRC)/fastxml.cpp ../tools/c++/mat.cpp $(LIBFLAGS)

PfastreXML_predict:
	$(CXX) -o pfastrexml_predict $(CXXFLAGS) $(INC) pfastrexml_predict.cpp $(SRC)/fastxml.cpp ../tools/c++/mat.cpp $(LIBFLAGS)

clean:
	rm -f pfastreXML_train pfastreXML_predict
