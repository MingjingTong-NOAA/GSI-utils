#!/bin/sh
#SBATCH -A gfdlhires
#SBATCH -o sort_out
#SBATCH -e sort_err
#SBATCH -q batch
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH -J sort_diag
#SBATCH --dependency=afterany:45077143
wrkdir=/scratch2/NCEPDEV/stmp3/Mingjing.Tong/corr_obs_iasi_metop-c/sea
ntot=124

cd $wrkdir
nt=1
one=1

while [[ $nt -le $ntot ]] ; do
   if [ $nt -lt 10 ] ; then
      fon=000$nt
   elif [ $nt -lt 100 ] ; then
      fon=00$nt
   elif [ $nt -lt 1000 ] ; then
      fon=0$nt
   else
      fon=$nt
   fi
   nd=$nt
   fonn=$fon
   while [ ! -f danl_${fonn} ] || [ ! -f dges_${fonn} ] ; do
      nd=$(( nd+one ))
      if [ $nd -lt 10 ] ; then
         fonn=000$nd
      elif [ $nd -lt 100 ] ; then
         fonn=00$nd
      elif [ $nd -lt 1000 ] ; then
         fonn=0$nd
      else
         fonn=$nd
      fi
      if [ $nd -gt $ntot ] ; then
         break
      fi
   done
   if [ $nd -gt $ntot ] ; then
      break
   fi
   if [ $nd -gt $nt ] ; then
      mv danl_${fonn} danl_${fon}
      mv dges_${fonn} dges_${fon}
   fi
   nt=$(( nt + one))
    
done
exit 0
