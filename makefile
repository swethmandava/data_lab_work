CC = g++
LIBS =

INCDIR = inc
OBJDIR = obj
SRCDIR = src
TARGETDIR = bin

EXE = test
TARGET = $(TARGETDIR)/$(EXE)

CFILES = $(wildcard $(SRCDIR)/*.cc)
_COBJECTS = $(CFILES:.cc=.o)
_DEPS = $(CFILES:.cc=.hh)

COBJECTS = $(patsubst $(SRCDIR)%.o,$(OBJDIR)%.o,$(_COBJECTS))
DEPS = $(patsubst $(SRCDIR)%.hh,$(INCDIR)%.hh,$(_DEPS))

CFLAGS = -O3 -std=c++11
INCLUDES = -I$(INCDIR)

default: all

all: $(TARGET)

obja: $(COBJECTS)

$(TARGET): $(COBJECTS) | $(TARGETDIR)
	$(CC) $(CFLAGS) -o $@ $(COBJECTS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDES)

obj:
	@mkdir -p $@

bin:
	@mkdir -p $@

.PHONY: clean all default

clean :
	rm -rf $(OBJDIR) $(SRCDIR)/*~ *~

cleanall:
	rm -rf $(OBJDIR) $(TARGETDIR) $(SRCDIR)/*~ *~
