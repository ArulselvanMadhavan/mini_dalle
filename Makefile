
all: format
	@dune build @all

format:
	@dune build @fmt --auto-promote
