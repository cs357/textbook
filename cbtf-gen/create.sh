#!/bin/bash
mkdir -p output
cd output/
wget --mirror --convert-links -nd https://cs357.github.io/textbook 
rm slides
wget --mirror --convert-links -nd https://cs357.github.io/textbook/slides
mv slides slides.html