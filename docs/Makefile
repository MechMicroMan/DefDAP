# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docs:
	@echo "Building API documentation to '/source/defdap'"
	sphinx-apidoc -feMT ../defdap -o ./source/defdap     
	@echo "Auto-generation of API documentation finished. " \
	"The generated files are in '/source/defdap'"
	@echo " "
	@echo "Cleaning build folder"
	make clean
	@echo " "
	@echo "Generating html documentation"
	make html
	@echo "html documentation built at /build/html"