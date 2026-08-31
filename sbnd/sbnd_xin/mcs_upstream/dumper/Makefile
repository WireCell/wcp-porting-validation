mcs_dump: mcs_dump.cxx ../mcs_standalone/src/mcs.h ../mcs_standalone/src/mcs.cxx
	`root-config --cxx --cflags` -I../mcs_standalone/src -o mcs_dump mcs_dump.cxx `root-config --libs`

# doc 83: links the INSTALLED toolkit port (ROOT-free), not the raw upstream
# source above -- this is the estimator that actually ran in the PR chain.
WCT_PREFIX = /nfs/data/1/xqian/toolkit-dev/local
mcs_probe: mcs_probe.cxx $(WCT_PREFIX)/include/WireCellMcs/MuonMCS.h
	g++ -std=c++17 -O2 -I$(WCT_PREFIX)/include -o mcs_probe mcs_probe.cxx \
	    -L$(WCT_PREFIX)/lib -Wl,-rpath,$(WCT_PREFIX)/lib -lWireCellMcs
