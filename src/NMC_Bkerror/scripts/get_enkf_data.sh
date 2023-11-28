#!/bin/ksh

set -x

sdate=2021061600
edate=2021071500
ndate=/home/Mingjing.Tong/bin/ndate

hpssdir=/NCEPDEV/emc-da/1year/Mingjing.Tong/HERA/scratch/sda_cntl
datadir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/NMC_Bkerror
NMEM_EARCGRP=10

cd $datadir
cdate=$sdate
while [[ $cdate -le $edate ]]; do
   ymd=`echo $cdate | cut -c1-8`
   hh=`echo $cdate | cut -c9-10`
   hpsstar get $hpssdir/$cdate/enkfgdas.tar ./enkfgdas.${ymd}/${hh}/atmos/gdas.t${hh}z.atmf006.ensmean.nc
   mv ./enkfgdas.${ymd}/${hh}/atmos/gdas.t${hh}z.atmf006.ensmean.nc $datadir/sfg_${cdate}_fhr06_ensmean
   rm -f ./enkfgdas.${ymd}/${hh}/atmos
   g=1
   while [[ $g -le 8 ]]; do
      m=1
      while [ $m -le $NMEM_EARCGRP ]; do
        nm=$(((g-1)*NMEM_EARCGRP+m))
        mem=$(printf %03i $nm)
        hpsstar get $hpssdir/$cdate/enkfgdas_grp0${g}.tar ./enkfgdas.${ymd}/${hh}/atmos/mem${mem}/gdas.t${hh}z.atmf006.nc
        mv ./enkfgdas.${ymd}/${hh}/atmos/mem${mem}/gdas.t${hh}z.atmf006.nc $datadir/sfg_${cdate}_fhr06_mem${mem}
        rm -rf ./enkfgdas.${ymd}/${hh}/atmos/mem${mem}
        m=$((m+1))
      done
      g=$((g+1))
   done
   cdate=`$ndate +24 $cdate`
done 

