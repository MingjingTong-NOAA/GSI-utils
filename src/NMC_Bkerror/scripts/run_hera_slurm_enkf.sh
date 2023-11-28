#!/bin/sh
#SBATCH --job-name=bkgnmc_enkf
#SBATCH --account=gfdlhires
#SBATCH --qos=batch
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=10
#SBATCH -t 05:00:00
#SBATCH -o %x.o%j

set -x

exp=C192_enkf_ens
base=/scratch2/GFDL/gfdlscr/Mingjing.Tong/GSI/GSI-utils

calstats=$base/install/bin/calcstats.x
sststats=$base/src/NMC_Bkerror/fix/sst2dvar_stat0.5.ufs

datdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/NMC_Bkerror

# Load modules
export MACHINE_ID=hera
source $base/ush/module-setup.sh
module use $base/modulefiles
module load gsiutils_hera.intel.lua
module list

tmpdir=/scratch1/NCEPDEV/stmp2/$LOGNAME/nmc/$exp
rm -rf $tmpdir
mkdir -p $tmpdir
cd $tmpdir

#export MPI_BUFS_PER_PROC=256
#export MPI_BUFS_PER_HOST=256
#export MPI_GROUP_MAX=256
#export OMP_NUM_THREADS=1
#export OMP_STACKSIZE=1024M
#export I_MPI_ADJUST_GATHERV=3
#export PSM2_MQ_RECVREQS_MAX=4000000

cp $calstats  ./stats.x
cp $sststats  ./berror_sst

ndate=/home/Mingjing.Tong/bin/ndate
date1='2021061600'
date2='2021071500'
date=$date1
/bin/rm -f infiles
touch infiles
# ens member files go first
while [ $date -le $date2 ]; do
  ls $datdir/sfg_${date}*mem* >> infiles
  date=`$ndate +24 $date`
done
# then corresponding ens mean files
date=$date1
while [ $date -le $date2 ]; do
  for filename in $datdir/sfg_${date}*mem*; do
     echo "$datdir/sfg_${date}_fhr06_ensmean" >> infiles
  done
  date=`$ndate +24 $date`
done

maxcases=`wc -l infiles | cut -f1 -d " "`

cat << EOF > stats.parm
 &NAMSTAT
   jcap=382,jcapin=382,jcapsmooth=382,nsig=91,nlat=386,nlon=768,maxcases=${maxcases},hybrid=.true.,smoothdeg=0.5,
   biasrm=.true.,vertavg=.true.,use_gfs_nemsio=.false.,use_gfs_ncio=.true.,use_enkf=.true.
 /
EOF

set -x
ln -s -f infiles fort.10

echo "I AM IN " $PWD

eval "srun $tmpdir/stats.x < stats.parm > nmcstats.out"

rc=$?

rm $tmpdir/fort.1*
rm $tmpdir/fort.2*
rm $tmpdir/fort.3*
rm $tmpdir/fort.4*
rm $tmpdir/fort.5*
rm $tmpdir/fort.6*
rm $tmpdir/fort.7*
rm $tmpdir/fort.8*
rm $tmpdir/fort.9*
rm $tmpdir/fort.0*

exit
