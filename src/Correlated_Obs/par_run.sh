

if [ ! -z "$NP" ] ; then
   export OMP_NUM_THREADS=$NP
fi
cdate=$bdate
nt=0
one=1
cd $wrkdir
while [[ $nt -le $ntot ]] ; do
   nt=$((nt + one))
   if [ $nt -lt 10 ] ; then
      fon=000$nt
   elif [ $nt -lt 100 ] ; then
      fon=00$nt
   elif [ $nt -lt 1000 ] ; then
      fon=0$nt
   else
      fon=$nt
   fi
   if [ ! -f dges_${fon} ];
   then
      nt=$((nt-one))
      break
   fi
   
done

./cov_calc <<EOF
$nt $type $cloud $angle $instr $wave_out $err_out $corr_out $npair_out $kreq $kreq2 $infdiag $infl $inflsurf $inflwv $method $cov_method $chan_set $time_sep $bsize $bcen $netcdf
EOF

set -x

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
mv Rcov_$instr Rcov_${instr}_${stype}
[ -f Rcorr_$instr ] && mv Rcorr_$instr Rcorr_${instr}_${stype}
[ -f wave_$instr ] && mv wave_$instr wave_${instr}_${stype}
[ -f err_$instr ] && mv err_$instr err_${instr}_${stype}
[ -f satinfo_err_$instr ] && mv satinfo_err_$instr satinfo_err_${instr}_${stype}
[ -f chnum_$instr ] && mv chnum_$instr chnum_${instr}_${stype}
[ -f npair_$instr ] && mv npair_$instr npair_${instr}_${stype}
[ -f eigs_$instr ] && mv eigs_$instr eigs_${instr}_${stype}

if [ ! -d $savdir/Rcov ]; then
  mkdir -p $savdir/Rcov
fi
if [ ! -d $savdir/Error ]; then
  mkdir -p $savdir/Error
fi

cp -f Rcov_${instr}_${stype} $savdir/Rcov/

[ -f Rcorr_${instr}_${stype} ] && cp -f Rcorr_${instr}_${stype} $savdir/Error/
[ -f wave_${instr}_${stype} ] && cp -f wave_${instr}_${stype} $savdir/Error/
[ -f err_${instr}_${stype} ] && cp -f err_${instr}_${stype} $savdir/Error/
[ -f satinfo_err_${instr}_${stype} ] && cp -f satinfo_err_${instr}_${stype} $savdir/Error/
[ -f chnum_${instr}_${stype} ] && cp -f chnum_${instr}_${stype} $savdir/Error/
[ -f npair_${instr}_${stype} ] && cp -f npair_${instr}_${stype} $savdir/Error/
[ -f eigs_${instr}_${stype} ] && cp -f eigs_${instr}_${stype} $savdir/Error/

exit 0
