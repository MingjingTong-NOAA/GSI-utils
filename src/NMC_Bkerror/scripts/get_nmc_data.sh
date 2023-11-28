#!/bin/sh
#SBATCH --job-name=get_data
#SBATCH --account=gfdlhires
#SBATCH --qos=batch
#SBATCH --partition=service
#SBATCH --nodes=1-1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 06:00:00
#SBATCH --mem=2048
#SBATCH -o /scratch2/GFDL/gfdlscr/Mingjing.Tong/GSI/GSI-utils/src/NMC_Bkerror/scripts/getdata.log
#SBATCH --export=NONE
#SBATCH --comment=7061b7bf9ac12e25e4f5a11e491ffcf1


set -x

ndate=/home/Mingjing.Tong/bin/ndate
datadir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/NMC_Bkerror

sdate=2020090100
edate=2021083000
hpssdir=/NCEPDEV/emc-da/1year/Mingjing.Tong/HERA/scratch/sfree4B

if [ 1 == 0 ]; then
cd $datadir
cdate=$sdate
while [[ $cdate -le $edate ]]; do
   ymd=`echo $cdate | cut -c1-8`
   hh=`echo $cdate | cut -c9-10`
   hpsstar get $hpssdir/$cdate/gfs_netcdfb.tar ./gfs.${ymd}/${hh}/atmos/gfs.t${hh}z.atmf048.nc
   mv ./gfs.${ymd}/${hh}/atmos/gfs.t${hh}z.atmf048.nc $datadir/${cdate}_atmf_48h.nc
   rm -rf ./gfs.${ymd}
   cdate=`$ndate +72 $cdate`
done 
fi

sdate=2021041500
edate=2021083100
hpssdir=/NCEPDEV/emc-da/1year/Mingjing.Tong/HERA/scratch/sfreeB24

cd $datadir
cdate=$sdate
while [[ $cdate -le $edate ]]; do
   ymd=`echo $cdate | cut -c1-8`
   hh=`echo $cdate | cut -c9-10`
   hpsstar get $hpssdir/$cdate/gfs_netcdfb.tar ./gfs.${ymd}/${hh}/atmos/gfs.t${hh}z.atmf024.nc
   mv ./gfs.${ymd}/${hh}/atmos/gfs.t${hh}z.atmf024.nc $datadir/${cdate}_atmf_24h.nc
   rm -rf ./gfs.${ymd}
   cdate=`$ndate +72 $cdate`
done

exit
