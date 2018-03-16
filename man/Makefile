#
# Makefile
#
# nnutil - Neural network utilities for tensorflow
# Copyright (c) 2018, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.


PREFIX       := /usr
MANDIR       := $(PREFIX)/share/man/man1

SOURCES      := $(wildcard *.rst)
HTML_TARGETS := $(patsubst %.rst,%.html,$(SOURCES))
MAN_TARGETS  := $(patsubst %.rst,%,$(SOURCES))

RST2HTML     := rst2html
RST2MAN      := rst2man

.PHONY: clean

all: man html

html: $(HTML_TARGETS)

$(HTML_TARGETS): %.html : %.rst
	$(RST2HTML) $? $@

man: $(MAN_TARGETS)

$(MAN_TARGETS): % : %.rst
	$(RST2MAN) $? $@

clean:
	-rm -f $(HTML_TARGETS)
	-rm -f $(MAN_TARGETS)
