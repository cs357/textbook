#!/bin/bash
mkdir -p output/
rm -rf output/
wget --mirror --convert-links http://127.0.0.1:4000/textbook/
mv 127.0.0.1:4000/textbook/ output/
echo "Build done! Upload the concents of the 'output/' directory *without modification* to PrairieLearn to be used in the CBTF."