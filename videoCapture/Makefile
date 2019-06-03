default: all

CPP = g++

CFLAGS = -Wall $(shell pkg-config --cflags opencv)
LDFLAGS = $(shell pkg-config --libs --static opencv)

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
EXEC = videoCapture

all: $(SRCS) $(EXEC)

%.o: %.cpp
	$(CPP) $(CFLAGS) -c -o $@ $<

$(EXEC): $(OBJS)
	$(CPP) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(OBJS) $(EXEC)
