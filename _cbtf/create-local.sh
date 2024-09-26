#!/bin/bash
set -e
mkdir -p output/
rm -rf output/
httrack http://127.0.0.1:4000/textbook/ -O output/ --stay-on-same-address -%v "-https://cs357.cs.illinois.edu*" +http://127.0.0.1:4000/textbook/* -c10 --disable-security-limits --mirror --max-rate=999999999
echo '<meta HTTP-EQUIV="Refresh" CONTENT="0; URL=127.0.0.1_4000/textbook/index.html">' > output/index.html
echo "Build done! Upload the concents of the 'output/' directory *without modification* to PrairieLearn to be used in the CBTF."