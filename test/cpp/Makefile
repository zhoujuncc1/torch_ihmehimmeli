CC=g++
CFLAGS=-I.
DEPS = test_functions.h
%.o: %.cc $(DEPS)
	$(CC) -g -c -o $@ $< $(CFLAGS)
all: testExponentiateSortedValidSpikes testActivateNeuronAlpha testLayer testCrossEntropyLoss

testExponentiateSortedValidSpikes: main_ExponentiateSortedValidSpikes.o functions.o 
	$(CC) -g -o testExponentiateSortedValidSpikes main_ExponentiateSortedValidSpikes.o functions.o 

testActivateNeuronAlpha: main_ActivateNeuronAlpha.o functions.o activateNeuronAlpha.o lambertw.o
	$(CC) -g -o testActivateNeuronAlpha main_ActivateNeuronAlpha.o functions.o activateNeuronAlpha.o lambertw.o

testLayer: main_Layer.o functions.o activateNeuronAlpha.o lambertw.o
	$(CC) -g -o testLayer main_Layer.o functions.o activateNeuronAlpha.o lambertw.o

testCrossEntropyLoss: main_CrossEntropyLoss.o functions.o 
	$(CC) -g -o testCrossEntropyLoss main_CrossEntropyLoss.o functions.o 

clean:
	rm *.o testExponentiateSortedValidSpikes testActivateNeuronAlpha testLayer testExponentiateActivation testLayer testCrossEntropyLoss
