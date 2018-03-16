#
# Makefile
#
# nnutil - Neural network utilities for tensorflow
# Copyright (c) 2018, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
