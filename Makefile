
all:
	@dune build @all

format:
	@dune build @fmt --auto-promote
