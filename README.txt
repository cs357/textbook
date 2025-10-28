**Development Quick Start**

* This repo has the modified RTD Jekyll theme as a submodule to keep licensing clean, clone with `git clone --recurse-submodules https://github.com/cs357/textbook` or `git submodule init && git submodule update`.

* You will need Node.JS since the Sass is compiled with a webpack pipeline. 
* You will need Ruby to run the Jekyll server.
* You will need `make` to run the scripts.

* `make install` to install dependencies.
* `make server` to start the development server - you will need to Ctrl-C and restart if there are CSS changes since they require re-compilation.
* `make build` to output the static HTML to the `_site/` folder.

You may use the image `ghcr.io/cs357/textbook-devel` as the development environment.

For example, run this to create a container with the Ruby dependencies

```
podman run \
       --volume $PWD:/srv/jekyll \
       --workdir /srv/jekyll \
       --rm \
       --interactive \
       --tty \
       --publish 4000:4000 \
       --publish 35729:35729 \
       ghcr.io/cs357/textbook-devel \
       /bin/bash
       sh -c 'make install && bundle exec jekyll server --verbose --incremental --host 0.0.0.0 --livereload --watch'
```
