#!/bin/bash

DATADIR="data"

echo "Installing prerequisites..."
pip install --user numpy matplotlib rootpy uncertainties

echo "Downloading data to directory '$DATADIR'..."
test ! -d "$DATADIR" && mkdir "$DATADIR"
test ! -e "$DATADIR/d0.root" && wget -P "$DATADIR" "http://web.physik.rwth-aachen.de/~hebbeker/fprakt/d0.root"
test ! -e "$DATADIR/mc_all.root" && wget -P "$DATADIR" "http://web.physik.rwth-aachen.de/~hebbeker/fprakt/mc_all.root"
