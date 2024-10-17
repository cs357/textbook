#!/bin/bash
mkdir -p output/
rm -rf output/
httrack https://cs357.cs.illinois.edu/textbook -O output/ --stay-on-same-address -%v -X "https://cs357.cs.illinois.edu/*" +https://cs357.cs.illinois.edu/textbook/*
echo '<meta HTTP-EQUIV="Refresh" CONTENT="0; URL=cs357.cs.illinois.edu/textbook/index.html">' > output/index.html
echo "Build done! Upload the concents of the 'output/' directory *without modification* to PrairieLearn to be used in the CBTF."
