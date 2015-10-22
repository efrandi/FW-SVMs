CXX? = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 1
MEB_LIB = MEB-based-SVMs.o MEB.o MEB-solvers.o MEB-SMO.o sCache.o FW-based-SVMs.o
SVM_LIB = svm.o CSVM-SMO.o NU-SVM-SMO.o Kernel.o Cache.o SVM-commons.o
UTILITIES = MEB-utilities.o random.o partitioner.o 
TRNG_INCLUDE = -I/lhome/jnancu/trng/include
TRNG_LIB = -L/lhome/jnancu/trng/lib -ltrng4

all: svm-train svm-predict svm-scale DL2-train DL2-predict 
lib: svm.o MEB-based-SVMs.o
	$(CXX) -shared svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES)
	$(CXX) $(CFLAGS) $^ -o svm-predict -lm

svm-train: svm-train.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES)
	$(CXX) $(CFLAGS) svm-train.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES) -o svm-train -lm

svm-scale: svm-scale.c
	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale

sync_experiment: sync_experiment.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES) rep_statistics.o sync_problem_generator.o
	$(CXX) $(CFLAGS) sync_experiment.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES) rep_statistics.o sync_problem_generator.o -o sync_experiment -lm $(TRNG_LIB)

DL2-train: DL2-train.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES)
	$(CXX) $(CFLAGS) DL2-train.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES) -o DL2-train -lm

DL2-predict: DL2-predict.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES)
	$(CXX) $(CFLAGS) DL2-predict.cpp $(SVM_LIB) $(MEB_LIB) $(UTILITIES) -o DL2-predict -lm

svm.o: svm.cpp svm.h Kernel.h CSVM-SMO.h NU-SVM-SMO.h MEB-based-SVMs.h SVM-commons.h
	$(CXX) $(CFLAGS) -c svm.cpp
CSVM-SMO.o: CSVM-SMO.cpp CSVM-SMO.h Kernel.h SVM-commons.h
	$(CXX) $(CFLAGS) -c CSVM-SMO.cpp
NU-SVM-SMO.o: NU-SVM-SMO.cpp NU-SVM-SMO.h CSVM-SMO.h SVM-commons.h
	$(CXX) $(CFLAGS) -c NU-SVM-SMO.cpp
Kernel.o: Kernel.cpp Kernel.h Cache.h SVM-commons.h
	$(CXX) $(CFLAGS) -c Kernel.cpp
Cache.o: Cache.cpp Cache.h SVM-commons.h
	$(CXX) $(CFLAGS) -c Cache.cpp
SVM-commons.o: SVM-commons.cpp SVM-commons.h 
	$(CXX) $(CFLAGS) -c SVM-commons.cpp
MEB-based-SVMs.o: MEB-based-SVMs.cpp MEB-based-SVMs.h MEB-utilities.h MEB-solvers.h SVM-commons.h random.h
	$(CXX) $(CFLAGS) -c MEB-based-SVMs.cpp	
MEB.o: MEB.cpp MEB.h MEB-utilities.h MEB-SMO.h SVM-commons.h
	$(CXX) $(CFLAGS) -c MEB.cpp 
MEB-solvers.o: MEB-solvers.cpp MEB-solvers.h MEB.h MEB-utilities.h MEB-kernels.h SVM-commons.h 
	$(CXX) $(CFLAGS) -c MEB-solvers.cpp 
MEB-SMO.o: MEB-SMO.cpp MEB-SMO.h MEB-utilities.h CSVM-SMO.h SVM-commons.h 
	$(CXX) $(CFLAGS) -c MEB-SMO.cpp 
FW-based-SVMs: FW-based-SVMs.cpp FW-based-SVMs.h MEB-SMO.h MEB-utilities.h
	$(CXX) $(CFLAGS) -c FW-based-SVMs.cpp 
sCache.o: sCache.cpp sCache.h SVM-commons.h  
	$(CXX) $(CFLAGS) -c sCache.cpp 
MEB-utilities.o: MEB-utilities.cpp MEB-utilities.h
	$(CXX) $(CFLAGS) -c MEB-utilities.cpp
random.o: random.cpp random.h
	$(CXX) $(CFLAGS) -c random.cpp
partitioner.o: partitioner.cpp partitioner.h random.h SVM-commons.h  
	$(CXX) $(CFLAGS) -c partitioner.cpp
rep_statistics.o: rep_statistics.cpp rep_statistics.h SVM-commons.h
	$(CXX) $(CFLAGS) -c rep_statistics.cpp
sync_problem_generator.o: sync_problem_generator.cpp sync_problem_generator.h 
	$(CXX) $(CFLAGS) $(TRNG_INCLUDE) -c sync_problem_generator.cpp
clean:
	rm -f *~ *.o svm-train svm-predict svm-scale DL2-train DL2-predict 
