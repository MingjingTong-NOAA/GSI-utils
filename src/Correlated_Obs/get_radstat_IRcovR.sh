#!/bin/ksh

set -x

sdate=2022062400
edate=2022082400
ndate=/home/Mingjing.Tong/bin/ndate

for exp in s2022_C192_v2p5_emdfdf; do
   datadir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/RadStat/shield_edmfdf
   hpssdir=/NCEPDEV/emc-da/1year/Mingjing.Tong/HERA/scratch/$exp

   if [[ ! -d $datadir ]]; then
      mkdir -p $datadir
   fi

   cd $datadir
   cdate=$sdate
   while [[ $cdate -le $edate ]]; do
      ymd=`echo $cdate | cut -c1-8`
      hh=`echo $cdate | cut -c9-10`
      if [[ ! -f ./radstat.gdas.${cdate} ]]; then
        hpsstar get $hpssdir/$cdate/gdas.tar ./gdas.$ymd/$hh/atmos/gdas.t${hh}z.radstat
        mv ./gdas.$ymd/$hh/atmos/gdas.t${hh}z.radstat ./radstat.gdas.${cdate}
        rm -rf ./gdas.$ymd
      fi
      cdate=`$ndate +6 $cdate`
   done
done
   
exit


