#!/bin/ksh

set -x

sdate=2021061500
edate=2021061500
ndate=/home/Mingjing.Tong/bin/ndate

hpssdir=/NCEPDEV/emc-da/1year/Mingjing.Tong/HERA/scratch/sda_cntl
datadir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/NMC_Bkerror
NMEM_EARCGRP=10

cd $datadir
cdate=$sdate
while [[ $cdate -le $edate ]]; do
   ymd=`echo $cdate | cut -c1-8`
   hh=`echo $cdate | cut -c9-10`
   mv sfg_${cdate}_fhr06_ensmean.nc sfg_${cdate}_fhr06_ensmean 
   g=1
   while [[ $g -le 8 ]]; do
      m=1
      while [ $m -le $NMEM_EARCGRP ]; do
        nm=$(((g-1)*NMEM_EARCGRP+m))
        mem=$(printf %03i $nm)
        mv sfg_${cdate}_fhr06_mem${mem}.nc sfg_${cdate}_fhr06_mem${mem}
        m=$((m+1))
      done
      g=$((g+1))
   done
   cdate=`$ndate +24 $cdate`
done 

