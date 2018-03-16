# nnutil - Neural network utilities for tensorflow
# Copyright (c) 2018, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.

NAME         := nnutil
VENVDIR      := venv

# Programs
PYTHON       := python3
LINT         := pylint
LINTARGS     := -E
VIRTUALENV   := virtualenv

# Virtual Environment
VPYTHON      := $(VENVDIR)/bin/$(notdir $(PYTHON))
VPIP         := $(VENVDIR)/bin/pip
VCOVERAGE    := $(VENVDIR)/bin/coverage
VPACKAGES    := coverage flake8

# Deployment paths
PREFIX       := /usr
MANDIR       := $(PREFIX)/share/man
DOCDIR       := $(PREFIX)/share/doc/$(NAME)
ZSHDIR       := $(PREFIX)/share/zsh/site-functions
BASHDIR      := /etc/bash_completion.d
SHEBANG      := /usr/bin/env $(PYTHON)

# Other variables
ARGS         :=

# Makefile
SHELL        := /usr/bin/bash
.SHELLFLAGS  := -e -u -c

SOURCE_FILES := $(shell find $(NAME)/ -name '*.py')


.ONESHELL:

# So we can use $$(variable) on the prerequisites, that expand at matching time.
.SECONDEXPANSION:



.PHONY: all build lint

all: build man

build:
	@$(PYTHON) setup.py build --executable="$(SHEBANG)"

lint:
	@$(LINT) $(LINTARGS) $(NAME) *.py



$(VENVDIR)/bin/activate: Makefile
	@$(VIRTUALENV) $(VENVDIR)
	$(VPIP) install $(VPACKAGES)

$(VENVDIR)/bin/$(NAME): $(VENVDIR)/bin/activate setup.py $(SOURCE_FILES)
	@$(VPIP) install --editable .



.PHONY: venv run test coverage

venv: $(VENVDIR)/bin/activate
	@$(SHELL) --init-file <(echo "source '$(HOME)/.bashrc'; source '$(VENVDIR)/bin/activate'")

run: $(VENVDIR)/bin/$(NAME)
	$(VPYTHON) $(VENVDIR)/bin/$(NAME) $(ARGS)

test: $(VENVDIR)/bin/activate
	@$(VPYTHON) -m unittest -v tests

coverage: $(VENVDIR)/bin/activate
	@$(VCOVERAGE) run -m unittest tests



.PHONY: clean

clean:
	@$(PYTHON) setup.py clean --all
	find . -name '*.pyc' -exec rm -f {} \;
	find . -name '.cache*' -exec rm -f {} \;
	find . -name '*.html' -exec rm -f {} \;
	rm -Rf $(VENVDIR) .coverage
	make -C man clean



.PHONY: man install update-template

man:
	@make -C man man

install:
	@$(PYTHON) setup.py install --prefix="$(PREFIX)" --root="$(DESTDIR)"
#	install -Dm644 "completion/zsh/_$(NAME)" "$(DESTDIR)$(ZSHDIR)/_$(NAME)"
#	install -Dm644 "completion/bash/$(NAME)" "$(DESTDIR)$(BASHDIR)/$(NAME)"
	install -d $(DESTDIR)$(MANDIR)/{man1,man5}
	install -Dm644 man/*.1 "$(DESTDIR)$(MANDIR)/man1/"
	install -Dm644 man/*.5 "$(DESTDIR)$(MANDIR)/man5/"


# -------------------------------------------------------------------------- #
# Source maintenance                                                         #
# -------------------------------------------------------------------------- #

.PHONY: update-template update-copyright

## Update cookiecutter template branch
update-template:
	@python make/cookiecutter-update.py ".cookiecutter.json" template

## Update copyright from file headers
update-copyright:
	@year=$$(date '+%Y')
	git ls-files | while read f; do
		sed -i "1,10{s/Copyright (c) \([0-9]\+\)\(-[0-9]\+\)\?,/Copyright (c) \1-$$year,/}" "$$f"
		sed -i "1,10{s/Copyright (c) $$year-$$year,/Copyright (c) $$year,/}" "$$f"
	done

.PHONY: help

## Print Makefile documentation
help:
	@perl -0 -nle 'printf("%-25s - %s\n", "$$2", "$$1") while m/^##\s*([^\r\n]+)\n^([\w-]+):[^=]/gm' \
		$(MAKEFILE_LIST) | sort
	printf "\n"
	perl -0 -nle 'printf("%-25s - %s\n", "$$2=", "$$1") while m/^##\s*([^\r\n]+)\n^([\w-]+)\s*:=/gm' \
		$(MAKEFILE_LIST) | sort
