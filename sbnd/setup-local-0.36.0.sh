# Standalone WCT 0.36.0 + larwirecell env, self-contained under
# /exp/sbnd/app/users/yuhw/wct-0.36.0-install.  Does NOT use the larsoft mrb
# build area, so it's unaffected by any larwirecell rebuild elsewhere.

source /cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh

setup sbndcode v10_14_02_03 -q e26:prof
setup gdb

# Activate our local larwirecell built into wct-0.36.0-install. We bypass
# `ups setup` because it fails on the larevt version conflict (sbndcode pulls
# v10_00_18, larwirecell.table requires v10_00_16), and we don't want UPS to
# re-pull cvmfs wirecell v0_32_1 anyway.
WCT036_INSTALL=/exp/sbnd/app/users/yuhw/wct-0.36.0-install
LWC_FQ="$WCT036_INSTALL/larwirecell/v10_01_28/slf7.x86_64.e26.prof"

# Drop the cvmfs larwirecell that sbndcode pulled in, so its libs/plugins
# don't shadow ours.
unsetup larwirecell 2>/dev/null

export LARWIRECELL_DIR="$WCT036_INSTALL/larwirecell/v10_01_28"
export LARWIRECELL_VERSION=v10_01_28
export LARWIRECELL_FQ_DIR="$LWC_FQ"
export LARWIRECELL_INC="$WCT036_INSTALL/larwirecell/v10_01_28/include"
export LARWIRECELL_LIB="$LWC_FQ/lib"
export LARWIRECELL_FCL="$WCT036_INSTALL/larwirecell/v10_01_28/fcl"

export PRODUCTS="$WCT036_INSTALL:$PRODUCTS"

# Path helpers (also defined in ~/.bashrc; redefined here so this script is
# usable in non-interactive contexts too).
path-remove ()
{
    local IFS=':';
    local NEWPATH;
    local DIR;
    local PATHVARIABLE=${2:-PATH};
    for DIR in ${!PATHVARIABLE};
    do
        if [ "$DIR" != "$1" ]; then
            NEWPATH=${NEWPATH:+$NEWPATH:}$DIR;
        fi;
    done;
    export $PATHVARIABLE="$NEWPATH"
}

path-prepend ()
{
    path-remove "$1" "$2";
    local PATHVARIABLE="${2:-PATH}";
    export $PATHVARIABLE="$1${!PATHVARIABLE:+:${!PATHVARIABLE}}"
}

path-append ()
{
    path-remove "$1" "$2";
    local PATHVARIABLE="${2:-PATH}";
    export $PATHVARIABLE="${!PATHVARIABLE:+${!PATHVARIABLE}:}$1"
}

# Point CMAKE_PREFIX_PATH, LD_LIBRARY_PATH, CET_PLUGIN_PATH, FHICL_FILE_PATH
# and PATH at the new WCT + larwirecell install. Order matters: larwirecell
# libs come first (so its WireCellToolkit module wins), then WCT.
path-prepend "$LWC_FQ/lib"              CET_PLUGIN_PATH
path-prepend "$LWC_FQ/lib"              LD_LIBRARY_PATH
path-prepend "$LARWIRECELL_FCL"         FHICL_FILE_PATH
path-prepend "$WCT036_INSTALL"          CMAKE_PREFIX_PATH
path-prepend "$WCT036_INSTALL/lib"      LD_LIBRARY_PATH
path-prepend "$WCT036_INSTALL/bin"      PATH

# WireCell config search path (unchanged from setup.sh).
path-prepend /exp/sbnd/app/users/yuhw/wire-cell-data WIRECELL_PATH
path-prepend /exp/sbnd/app/users/yuhw/sbndcode/sbndcode/WireCell/cfg WIRECELL_PATH

export PS1=(wct036)'$(pwd)\n$'

source /exp/sbnd/app/users/yuhw/wire-cell-python/venv/bin/activate
