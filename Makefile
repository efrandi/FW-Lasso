CXX? = g++ 
CFLAGS = -Wall -Wconversion -O3 -fPIC 
FW_LIB = FW_based_LASSO.o LASSO_kernel.o sCache.o

all: lasso-train

lasso-train: LASSO_train.h LASSO_train.cpp $(FW_LIB)
	$(CXX) $(CFLAGS) LASSO_train.cpp $(FW_LIB) -o lasso-train -lm

FW_based_LASSO.o: FW_based_LASSO.cpp FW_based_LASSO.h LASSO_definitions.h
	$(CXX) $(CFLAGS) -c FW_based_LASSO.cpp
LASSO_kernel.o: LASSO_kernel.cpp LASSO_kernel.h sCache.h LASSO_definitions.h
	$(CXX) $(CFLAGS) -c LASSO_kernel.cpp
sCache.o: sCache.cpp sCache.h LASSO_definitions.h
	$(CXX) $(CFLAGS) -c sCache.cpp

clean:
	rm -f *~ *.o lasso-train 
