#!/bin/bash
mkdir -p output/
rm -rf output/
httrack https://cs357.github.io/textbook/ -O output/
echo '<meta HTTP-EQUIV="Refresh" CONTENT="0; URL=cs357.github.io/textbook/index.html">' > output/index.html