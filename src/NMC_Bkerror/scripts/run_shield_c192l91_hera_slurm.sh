#!/bin/ksh --login
#SBATCH --job-name=bkgnmc
#SBATCH --account=gfdlhires
#SBATCH --qos=batch
#SBATCH --nodes=30
#SBATCH --ntasks-per-node=10
#SBATCH -t 03:00:00
#SBATCH -o %x.o%j

set -x

exp=C192_nmc_00z_scale5
base=/scratch2/GFDL/gfdlscr/Mingjing.Tong/global_workflow/shield_develop/sorc/gsi.fd

calstats=$base/util/NMC_Bkerror/sorc/calcstats.exe
sststats=$base/util/NMC_Bkerror/fix/sst2dvar_stat0.5.ufs
scalefile=$base/util/NMC_Bkerror/scaling/berror_l91_scaling.txt

datdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/DABERROR/C192

set -x

tmpdir=/scratch2/NCEPDEV/stmp1/$LOGNAME/nmc/$exp
rm -rf $tmpdir
mkdir -p $tmpdir
cd $tmpdir

export MPI_BUFS_PER_PROC=256
export MPI_BUFS_PER_HOST=256
export MPI_GROUP_MAX=256
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1024M
export I_MPI_ADJUST_GATHERV=3
export PSM2_MQ_RECVREQS_MAX=4000000

module load intel
module load impi

cp $calstats  ./stats.x
cp $sststats  ./berror_sst
cp $scalefile ./scaling.txt

ls $datdir/*00_atmf_24h.nemsio >> infiles
ls $datdir/*00_atmf_48h.nemsio >> infiles
#ls $datdir/*06_atmf_24h.nemsio >> infiles
#ls $datdir/*06_atmf_48h.nemsio >> infiles
#ls $datdir/*24h.nemsio >> infiles
#ls $datdir/*48h.nemsio >> infiles

maxcases=`wc -l infiles | cut -f1 -d " "`

cat << EOF > stats.parm
 &NAMSTAT
   jcap=382,jcapin=382,jcapsmooth=382,nsig=91,nlat=386,nlon=768,maxcases=${maxcases},hybrid=.true.,smoothdeg=0.5,
   biasrm=.true.,vertavg=.true.,use_gfs_nemsio=.true.,use_enkf=.false.,scaling=.true.,
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
