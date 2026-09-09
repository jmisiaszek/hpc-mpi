CXX         = CC
CXXFLAGS    = -O3 -Wall -std=c++2a

DEPS    = ilu.h
OBJ     = ilu.o
BINS    = example_test.my test.my test_ign.my
IGN     = ign.o
OLD     = old.o

IGN_BINS = example_test.ign test.ign test_ign.ign
OLD_BINS = example_test.old test.old test_ign.old

all: $(BINS)
ign: $(IGN_BINS)
old: $(OLD_BINS)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $

%.my: %.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.ign: %.o $(IGN)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.old: %.o $(OLD)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: all ign old clean

clean:
	rm -f *.o *.out *.err $(BINS) $(IGN_BINS) $(OLD_BINS)