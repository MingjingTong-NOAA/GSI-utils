#! /usr/bin/env bash
set -eux

#source ./machine-setup.sh > /dev/null 2>&1
#cwd=`pwd`

./configure hera

source ./modulefile.nmcberror.hera

make -f Makefile clean
make -f Makefile

