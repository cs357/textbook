#!/bin/bash
mkdir -p output/
rm -rf output/
httrack https://cs357.cs.illinois.edu/textbook -O output/ -*textbook/links*
echo '<meta HTTP-EQUIV="Refresh" CONTENT="0; URL=cs357.github.io/textbook/index.html">' > output/index.html
echo "Build done! Upload the concents of the 'output/' directory *without modification* to PrairieLearn to be used in the CBTF."
