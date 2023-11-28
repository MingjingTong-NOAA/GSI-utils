#!/bin/sh
#date of first radstat file
bdate=2022070100
#date of last radstat file
edate=2022073018
#instrument name, as it would appear in the title of a diag file
#instr=airs_aqua
instr=iasi_metop-b
#location of radstat file
#exp=s2022_C192_v2p5_edmf1ens0
exp=s2022_C192_edmf0ens1_RcovdiasiR1p5_QCold
#exp=s2022_C192_edmf0ens1_RcovdiasiR1p75_QCold
diagdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/scrub/RadDiag/${exp}
#working directory
wrkdir=/scratch2/NCEPDEV/stmp3/${USER}/corr_obs_${instr}
newrun="YES"
#location the covariance matrix is saved to
savdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/noscrub/Rcov/cloud1
#FOV type- 0 for all, 1 for sea, 2 for land, 3 for snow, 
#4 for mixed (recommended to use 0 for mixed)
#5 for ice and 6 for snow and ice combined (recommended when using ice)
type=1
#cloud 1 for clear FOVs, 2 for clear channels
cloud=1
#absolute value of the maximum allowable sensor zenith angle (degrees)
angle=30
#option to output the channel wavenumbers
wave_out=.true.
#option to output the observation errors
err_out=.true.
#option to output the correlation matrix
corr_out=.true.
#option to output the observation pairs
npair_out=.true.
#condition number to recondition Rcov.  Set <0 to not recondition
#kreq=200
kreq=-1
#condition number to recondition Rcov (< 1.0)
kreq2=0.0
#inflation factors, for regular channels, surface channels and water vapor channels
#infl is applied to all channels if using binary files or a MW instrument
#set factors equal to 1 to not inflate, or if this channel group is not assimilated
infdiag=.false.
infl=1.0
inflsurf=1.0
inflwv=1.0
#method to recondition:  1 for trace method, 2 for Weston's second method
method=1
#method to compute covariances: 1 for Hollingsworth-Lonnberg, 2 for Desroziers
cov_method=2
#maximum time between observations in a pair, in minutes
time_sep=1.0
#bin size for obs pairs in km
bsize=1
#bin center, in km, needed for Hollingsworth-Lonnberg
bcen=80
if [ $cov_method -eq 1 ]; then
  wrkdir=/scratch2/NCEPDEV/stmp3/${USER}/corr_obs_${instr}_hl
  if [ $infdiag = ".true." ]; then
    savdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/noscrub/Rcov/${exp}/cloud${cloud}_hl_bcen${bcen}_kreq${kreq}_inf${infl}
  else
    savdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/noscrub/Rcov/${exp}/cloud${cloud}_hl_bcen${bcen}_kreq${kreq}_infR${infl}k${kreq2}
  fi
else
  wrkdir=/scratch2/NCEPDEV/stmp3/${USER}/corr_obs_${instr}
  if [ $infdiag = ".true." ]; then
    savdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/noscrub/Rcov/${exp}/cloud${cloud}_kreq${kreq}_inf${infl}
  else
    savdir=/scratch2/GFDL/gfdlscr/Mingjing.Tong/noscrub/Rcov/${exp}/cloud${cloud}_kreq${kreq}_infR${infl}k${kreq2}
  fi
fi
#channel set choice:  0 to only use active channels, 1 to use all channels
chan_set=0
#number of processors to use to unpack radstat files-most efficient if # of radstats/$num_proc has a small remainder
num_proc=40
#number of processors to run cov_calc on
NP=10
#wall time to unpack radstat files format hh:mm:ss for hera, hh:mm for wcoss
unpack_walltime=02:30:00
#wall time to run cov_calc hh:mm:ss for hera, hh:mm for wcoss
wall_time=01:00:00
#requested memory in MB to unpack radstats, on WCOSS/Cray.  Increases with decreasing $num_proc
#should be at least 15
Umem=50
#requested memory in MB for cov_calc, on WCOSS/Cray
Mem=50
#job account name (needed on hera only)
account=gfdl-hires
#job project code (needed on wcoss only)
project_code=GFS-T2O
#machine-hera or wcoss, all lower case
machine=hera
#netcdf or binary diag files-0 for binary, 1 for netcdf
netcdf=1
ndate=/home/Mingjing.Tong/bin/ndate

####################################################################

stype=sea
if [ $type -eq 0 ] ; then
   stype=glb
elif [ $type -eq 2 ] ; then
   stype=land
elif [ $type -eq 3 ] ; then
   stype=snow
elif [ $type -eq 4 ] ; then
   stype=mixed
elif [ $type -eq 5 ] ; then
   stype=ice
elif [ $type -eq 6 ] ; then
   stype=snow_ice
fi
wrkdir=${wrkdir}/${stype}
cdate=$bdate
if [[ -d ${wrkdir} && $newrun == "YES" ]]; then
  rm -rf ${wrkdir}
fi
if [ ! -d ${wrkdir} ]; then
  mkdir -p ${wrkdir}
fi


nt=0
one=1
while [[ $cdate -le $edate ]] ; do
   while [[ ! -f $diagdir/radstat.gdas.$cdate ]] ; do 
      cdate=`$ndate +06 $cdate`
      if [ $cdate -ge $edate ] ; then
         break
      fi
   done
   cdate=`$ndate +06 $cdate`
   nt=$((nt+one))
done
dattot=$nt
cp unpack_rads.sh $wrkdir
cp par_run.sh $wrkdir
cp sort_diags.sh $wrkdir
cp ../../install/bin/cov_calc.x $wrkdir/cov_calc

cd $wrkdir
if [ $newrun = "YES" ]; then
num_jobs=$num_proc
if [ $num_proc -ge $nt ] ; then
   num_jobs=$nt
fi
jobs_per_proc=$((nt/num_jobs))
last_proc_jobs=$((nt-jobs_per_proc*num_jobs+jobs_per_proc))
nt=1
cdate=$bdate
while [[ $nt -le $num_jobs ]] ; do
   nd=1
   date1=$cdate
   jobsn=$jobs_per_proc
   if [ $nt -eq $num_jobs ] ; then
      jobsn=$last_proc_jobs
   fi
   while [[ $nd -lt $jobsn ]] ; do
      while [[ ! -f $diagdir/radstat.gdas.$cdate ]] ; do
         cdate=`$ndate +06 $cdate`
         if [ $cdate -gt $edate ] ; then
            break
         fi
      done
      nd=$((nd + one))
      cdate=`$ndate +06 $cdate`
   done
   date2=$cdate
   numin=$((nt - one))
   coun=1
   if [[ $numin -gt 0 ]] ; then
      numin=$((numin*jobs_per_proc))
   fi
   numin=$((numin + one))
    
   cat << EOF > params.sh
#!/bin/sh
bdate=$date1
edate=$date2
start_nt=$numin
ndate=$ndate
wrkdir=$wrkdir
diagdir=$diagdir
instr=$instr
netcdf=$netcdf
EOF
   chmod +rwx params.sh
   cat unpack_rads.sh >> params.sh
   mv params.sh unpack_rads_${nt}.sh
   nt=$((nt + one))
   cdate=`$ndate +06 $cdate`
done

cat << EOF > jobchoice.sh
#!/bin/sh
nt=\$1
one=1
njobs=$jobs_per_proc
./unpack_rads_\${nt}.sh
EOF
chmod +rwx jobchoice.sh

if [ $machine = hera ] ; then
cat << EOF > jobarray.sh
#!/bin/sh
#SBATCH -A $account
#SBATCH -o unpack_out
#SBATCH -e unpack_err
#SBATCH -q batch
#SBATCH --time=${unpack_walltime}
#SBATCH --ntasks=1
#SBATCH -J unpack
#SBATCH --array 1-${num_jobs}
cd $wrkdir
./jobchoice.sh \${SLURM_ARRAY_TASK_ID}
EOF
jobid=$(sbatch jobarray.sh)
elif [ $machine = wcoss ] ; then
cat << EOF > jobarray.sh
#!/bin/sh
#BSUB -o unpack_out
#BSUB -e unpack_err
#BSUB -q dev
#BSUB -M ${Umem}
#BSUB -n 1
#BSUB -W ${unpack_walltime}
#BSUB -R span[ptile=1]
#BSUB -P ${project_code}
#BSUB -J unpack[1-${num_jobs}]
cd $wrkdir
echo ${LSB_JOBINDEX}
./jobchoice.sh \${LSB_JOBINDEX} 
EOF
bsub < jobarray.sh
else
   echo cannot submit job, not on hera or wcoss
   exit 1
fi
#check if shifts are needed
if [ $machine = hera ] ; then
cat << EOF > params.sh
#!/bin/sh
#SBATCH -A $account
#SBATCH -o sort_out
#SBATCH -e sort_err
#SBATCH -q batch
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH -J sort_diag
#SBATCH --dependency=afterany:${jobid##* }
wrkdir=$wrkdir
ntot=$dattot
EOF
chmod +rwx params.sh
cat sort_diags.sh >> params.sh
mv params.sh sort_diags.sh

jobid=$(sbatch sort_diags.sh )
elif [ $machine = wcoss ] ; then
cat << EOF > params.sh
#!/bin/sh
#BSUB -o sort_out
#BSUB -e sort_err
#BSUB -q dev
#BSUB -M 30
#BSUB -n 1
#BSUB -W 00:02
#BSUB -R span[ptile=1]
#BSUB -P ${project_code}
#BSUB -J sort_diag
wrkdir=$wrkdir
ntot=$dattot
EOF
chmod +rwx params.sh
cat sort_diags.sh >> params.sh
mv params.sh sort_diags.sh
bsub -w "done(unpack)" < sort_diags.sh
#bsub < sort_diags.sh
else
   exit 1
fi
fi

#run cov_calc
if [ $machine = hera ] ; then
cat > params.sh << EOF
#!/bin/sh
#SBATCH -A $account
#SBATCH -o comp_out
#SBATCH -e comp_err
#SBATCH -q batch
#SBATCH --time=$wall_time
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$NP
#SBATCH -J cov_calc
EOF
if [ $newrun = "YES" ]; then
  cat >> params.sh << EOF
#SBATCH --dependency=after:${jobid##* }
EOF
else
  rm -f comp_err comp_out
fi
cat >> params.sh << EOF
bdate=$bdate
edate=$edate
instr=$instr
diagdir=$diagdir
wrkdir=$wrkdir
savdir=$savdir
type=$type
cloud=$cloud
angle=$angle
wave_out=$wave_out
err_out=$err_out
corr_out=$corr_out
npair_out=$npair_out
kreq=$kreq
kreq2=$kreq2
infdiag=$infdiag
infl=$infl
inflsurf=$inflsurf
inflwv=$inflwv
method=$method
cov_method=$cov_method
time_sep=$time_sep
bsize=$bsize
bcen=$bcen
chan_set=$chan_set
ntot=$dattot
NP=$NP
netcdf=$netcdf
EOF
chmod +rwx params.sh
cat par_run.sh >> params.sh
mv params.sh par_run.sh
sbatch par_run.sh
elif [ $machine = wcoss ] ; then
cat << EOF > params.sh
#!/bin/sh
#BSUB -o comp_out
#BSUB -e comp_err
#BSUB -openmp
#BSUB -q dev
#BSUB -M ${Mem}
#BSUB -n $NP
#BSUB -W $wall_time
#BSUB -R span[ptile=$NP]
#BSUB -P ${project_code}
#BSUB -J cov_calc
bdate=$bdate
edate=$edate
instr=$instr
diagdir=$diagdir
wrkdir=$wrkdir
savdir=$savdir
type=$type
cloud=$cloud
angle=$angle
wave_out=$wave_out
err_out=$err_out
corr_out=$corr_out
npair_out=$npair_out
kreq=$kreq
kreq2=$kreq2
infdiag=$infdiag
infl=$infl
inflsurf=$inflsurf
inflwv=$inflwv
method=$method
cov_method=$cov_method
time_sep=$time_sep
bsize=$bsize
netcdf=$netcdf
bcen=$bcen
chan_set=$chan_set
ntot=$dattot
NP=$NP
EOF
chmod +rwx params.sh
cat par_run.sh >> params.sh
mv params.sh par_run.sh
bsub -w "done(sort_diag)" < par_run.sh
else 
   exit 1
fi
exit 0
