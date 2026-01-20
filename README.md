# CS 357 Textbook

## Overview

* This repo has the modified RTD Jekyll theme as a submodule to keep licensing clean, clone with `git clone --recurse-submodules https://github.com/cs357/textbook` or `git submodule init && git submodule update`.

* You will need Node.JS since the Sass is compiled with a webpack pipeline. 
* You will need Ruby to run the Jekyll server.
* You will need `make` to run the scripts.
* We reccomend that you use the development container configuration included in this repository.

## Getting Started

* `make install` to install dependencies.
* `make server` to start the development server - you will need to Ctrl-C and restart if there are CSS changes since they require re-compilation.
* `make build` to output the static HTML to the `_site/` folder.