program getgribncensmeanp
!$$$  main program documentation block
!
! program:  getgribncensmeanp              compute ensemble mean
!
! prgmmr: whitaker         org: esrl/psd               date: 2009-02-23
!
! abstract:  create ensemble mean NCEP GFS grib nc file.
!
! program history log:
!   2009-02-23  Initial version.
!
! usage:
!   input files:
!
!   output files:
!
! attributes:
!   language: f95
!
!$$$

  use netcdf
  use module_ncio, only: open_dataset, create_dataset, read_attribute, &
                         Dataset, Dimension, close_dataset, has_attr, &
                         read_vardata, write_attribute, write_vardata, &
                         get_dim, quantize_data, has_var

  implicit none

  real,parameter :: zero=0.0_4
  integer,parameter :: iunit=21

  logical :: lexist,quantize,write_spread_ncio
  character(len=3) :: charnanal
  character(len=500) :: filenamein,filenameout,filenameouts,datapath,fileprefix,fname,&
                        filenameoutsprd
  integer :: iret,nlevs,nanals,k,ndims,nvar,nbits
  integer :: mype,mype1,npe,orig_group,new_group,new_comm
  integer :: latb,lonb,n
  integer,allocatable,dimension(:) :: new_group_members
  real(8) :: rnanals,rnanalsm1,t1,t2
  real(4),allocatable, dimension(:,:,:) :: values_3d, values_3d_avg, &
                                           values_3d_tmp, values_3d_sprd
  real(4),allocatable, dimension(:,:,:) :: values_3dv, values_3dv_avg, &
                                           values_3dv_tmp, values_3dv_sprd
  real(4),allocatable,dimension(:,:) :: values_2d, values_2d_sprd, &
                                        values_2d_avg, values_2d_tmp
  real(4) compress_err
  real(4),allocatable,dimension(:) :: values_1d, values_1d_avg, leveldiff


  type(Dataset) :: dset,dseto,dseto_sprd
  type(Dimension) :: londim,latdim,levdim

! mpi definitions.
  include 'mpif.h'

! Initialize mpi, mype is process number, npe is total number of processes.
  call mpi_init(iret)
  call mpi_comm_rank(mpi_comm_world,mype,iret)
  call mpi_comm_size(mpi_comm_world,npe,iret)

  mype1 = mype + 1

  if ( mype == 0 ) call w3tagb('GETSIGENSMEAN_SMOOTH',2011,0319,0055,'NP25')

! Get user input from command line
  call getarg(1,datapath)
  call getarg(2,filenameout)
  call getarg(3,fileprefix)
  call getarg(4,charnanal)
  read(charnanal,'(i3)') nanals

  rnanals = nanals
  rnanals = 1.0_8/rnanals
  rnanalsm1 = nanals-1
  rnanalsm1 = 1.0_8/rnanalsm1
  filenameout = trim(adjustl(datapath)) // trim(adjustl(filenameout))
  ! if a 5th arg present, it's a filename to write out ensemble spread
  write_spread_ncio = .true.
  if (iargc() > 4) then
     call getarg(5,filenameoutsprd)
     write_spread_ncio = .true.
     if (mype == 0) print *,'computing ensemble spread'
     filenameoutsprd = trim(adjustl(datapath)) // trim(adjustl(filenameoutsprd))
  endif

  if ( mype == 0 ) then
     write(6,'(a)')  'Command line input'
     write(6,'(a,a)')' datapath    = ',trim(datapath)
     write(6,'(a,a)')' filenameout = ',trim(filenameout)
     write(6,'(a,a)')' fileprefix  = ',trim(fileprefix)
     write(6,'(a,a)')' nanals      = ',trim(charnanal)
     if (write_spread_ncio) then
     write(6,'(a,a)')' filenameoutsprd = ',trim(filenameoutsprd)
     endif
     write(6,'(a)')  ' '
  endif

  if ( npe < nanals ) then
     write(6,'(2(a,i4))')'***FATAL ERROR***  npe too small.  npe = ',npe,' < nanals = ',nanals
     call mpi_abort(mpi_comm_world,99,iret)
     stop
  end if

! Create sub-communicator to handle number of cases (nanals)
  call mpi_comm_group(mpi_comm_world,orig_group,iret)

  allocate(new_group_members(nanals))
  do k=1,nanals
     new_group_members(k)=k-1
  end do

  call mpi_group_incl(orig_group,nanals,new_group_members,new_group,iret)
  call mpi_comm_create(mpi_comm_world,new_group,new_comm,iret)
  if ( iret /= 0 ) then
     write(6,'(a,i5)')'***FATAL ERROR*** after mpi_comm_create with iret = ',iret
     call mpi_abort(mpi_comm_world,101,iret)
  endif

! Process input files (one file per task)
  if ( mype1 <= nanals ) then

     write(charnanal,'(i3.3)') mype1
     filenamein = trim(adjustl(datapath)) // &
          trim(adjustl(fileprefix)) // '_mem' // charnanal

     dset = open_dataset(filenamein,errcode=iret)

!    Read each ensemble member
     if (mype == 0) write(6,*) 'Read netcdf'
     londim = get_dim(dset,'longitude'); lonb = londim%len
     latdim = get_dim(dset,'latitude'); latb = latdim%len
     levdim = get_dim(dset,'level');   nlevs = levdim%len

     if ( mype == 0 ) then
        write(6,'(a)')   ' '
        write(6,'(2a)')  'Read header information from ',trim(filenamein)
        write(6,'(a,i9)')' nlevs   = ',nlevs
        write(6,'(a,i9)')' lonb    = ',lonb
        write(6,'(a,i9)')' latb    = ',latb
        write(6,'(a)')   ' '
     endif

!     if (ncio) then
        if (mype == 0) then
           t1 = mpi_wtime()
           dseto = create_dataset(filenameout, dset, copy_vardata=.true.)
           if (write_spread_ncio) then
              dseto_sprd = create_dataset(filenameoutsprd, dset, copy_vardata=.true.)
           endif
        endif

        allocate(values_1d_avg(nlevs), leveldiff(nlevs))
        call read_vardata(dset,'level',values_1d)
        call mpi_allreduce(values_1d,values_1d_avg,nlevs,mpi_real4,mpi_sum,new_comm,iret)
        values_1d_avg = values_1d_avg*rnanals
        leveldiff = values_1d - values_1d_avg
        if ( sum(leveldiff) /= 0. ) then
           write(6,'(2(a,i4))')'***FATAL ERROR***  levels are not consisitent'
           call mpi_abort(mpi_comm_world,99,iret)
           stop
        end if
        if (allocated(values_1d)) deallocate(values_1d)
        deallocate(values_1d_avg,leveldiff)

        allocate(values_2d_avg(lonb,latb))
        allocate(values_2d_tmp(lonb,latb))
        if (write_spread_ncio) allocate(values_2d_sprd(lonb,latb))
        do nvar=1,dset%nvars
           ndims = dset%variables(nvar)%ndims
           if (ndims > 2) then
               if (ndims == 3 .and. trim(dset%variables(nvar)%name) /= 'hgtsfc') then
                  ! pressfc
                  if (mype == 0) print *,'processing ',trim(dset%variables(nvar)%name)
                  call read_vardata(dset,trim(dset%variables(nvar)%name),values_2d)
                  call mpi_allreduce(values_2d,values_2d_avg,lonb*latb,mpi_real4,mpi_sum,new_comm,iret)
                  ! ens mean
                  values_2d_avg = values_2d_avg*rnanals
                  if (write_spread_ncio) then
                     ! ens spread
                     values_2d_tmp = values_2d - values_2d_avg ! ens pert
                     values_2d_tmp = values_2d_tmp**2
                     call mpi_reduce(values_2d_tmp,values_2d_sprd,lonb*latb,mpi_real4,mpi_sum,0,new_comm,iret)
                     values_2d_sprd= sqrt(values_2d_sprd*rnanalsm1)
                     if (mype == 0) print *,trim(dset%variables(nvar)%name),' min/max spread',minval(values_2d_sprd),maxval(values_2d_sprd)
                  endif
                  if (has_attr(dset, 'nbits', trim(dset%variables(nvar)%name))) then
                      call read_attribute(dset, 'nbits', nbits, &
                           trim(dset%variables(nvar)%name))
                      quantize = .true.
                      if (nbits < 1) quantize = .false.
                  else
                      quantize = .false.
                  endif
                  ! write ens mean
                  if (mype == 0) then
                     if (quantize) then
                       values_2d_tmp = values_2d_avg
                       call quantize_data(values_2d_tmp, values_2d_avg, nbits, compress_err)
                       call write_attribute(dseto,&
                       'max_abs_compression_error',compress_err,trim(dset%variables(nvar)%name))
                     endif
                     call write_vardata(dseto,trim(dset%variables(nvar)%name),values_2d_avg)
                     if (write_spread_ncio) then
                        if (quantize) then
                          values_2d_tmp = values_2d_sprd
                          call quantize_data(values_2d_tmp, values_2d_sprd, nbits, compress_err)
                          call write_attribute(dseto_sprd,&
                          'max_abs_compression_error',compress_err,trim(dset%variables(nvar)%name))
                        endif
                        call write_vardata(dseto_sprd,trim(dset%variables(nvar)%name),values_2d_sprd)
                     endif
                  endif
               else if (ndims == 4) then
                  ! 3d variables (extra dim is time)
                  call read_vardata(dset,trim(dset%variables(nvar)%name),values_3d)
                  if (allocated(values_3d_avg)) deallocate(values_3d_avg)
                  allocate(values_3d_avg, mold=values_3d)
                  if (allocated(values_3d_tmp)) deallocate(values_3d_tmp)
                  allocate(values_3d_tmp, mold=values_3d_avg)
                  if (write_spread_ncio) then
                     if (allocated(values_3d_sprd)) deallocate(values_3d_sprd)
                     allocate(values_3d_sprd, mold=values_3d_avg)
                  endif
                  if (mype == 0) print *,'processing ',trim(dset%variables(nvar)%name)
                  call mpi_allreduce(values_3d,values_3d_avg,lonb*latb*nlevs,mpi_real4,mpi_sum,new_comm,iret)
                  values_3d_avg = values_3d_avg*rnanals
                  if (write_spread_ncio) then
                     ! ens spread
                     values_3d_tmp = values_3d - values_3d_avg ! ens pert
                     if (mype == 0 .and. trim(dset%variables(nvar)%name) == 't') then
                        !if (mype == 0) print *,trim(dset%variables(nvar)%name),' min/max mem', minval(values_3d),maxval(values_3d)
                        !if (mype == 0) print *,trim(dset%variables(nvar)%name),' min/max avg', minval(values_3d_avg),maxval(values_3d_avg)
                        !if (mype == 0) print *,trim(dset%variables(nvar)%name),' min/max diff', minval(values_3d_tmp),maxval(values_3d_tmp) 
                        print *,'member ', values_3d(34,40,1:20)
                        print *,'mean ', values_3d_avg(34,40,1:20)
                        print *,'diff ', values_3d_tmp(34,40,1:20)
                     end if
                     values_3d_tmp = values_3d_tmp**2
                     call mpi_reduce(values_3d_tmp,values_3d_sprd,lonb*latb*nlevs,mpi_real4,mpi_sum,0,new_comm,iret)
                     values_3d_sprd= sqrt(values_3d_sprd*rnanalsm1)
                     if (mype == 0) print *,trim(dset%variables(nvar)%name),' min/max spread',minval(values_3d_sprd),maxval(values_3d_sprd)
                  endif
                  if (has_attr(dset, 'nbits', trim(dset%variables(nvar)%name))) then
                      call read_attribute(dset, 'nbits', nbits, &
                           trim(dset%variables(nvar)%name))
                      quantize = .true.
                      if (nbits < 1) quantize = .false.
                  else
                      quantize = .false.
                  endif
                  if (mype == 0) then
                     if (quantize) then
                       values_3d_tmp = values_3d_avg
                       call quantize_data(values_3d_tmp, values_3d_avg, nbits, compress_err)
                       call write_attribute(dseto,&
                       'max_abs_compression_error',compress_err,trim(dset%variables(nvar)%name))
                     endif
                     call write_vardata(dseto,trim(dset%variables(nvar)%name),values_3d_avg)
                     if (write_spread_ncio) then
                        if (quantize) then
                          values_3d_tmp = values_3d_sprd
                          call quantize_data(values_3d_tmp, values_3d_sprd, nbits, compress_err)
                          call write_attribute(dseto_sprd,&
                          'max_abs_compression_error',compress_err,trim(dset%variables(nvar)%name))
                        endif
                        call write_vardata(dseto_sprd,trim(dset%variables(nvar)%name),values_3d_sprd)
                     endif
                  endif
               endif
           endif ! ndims > 2
        enddo  ! nvars
        if (allocated(values_2d)) deallocate(values_2d)
        if (allocated(values_3d)) deallocate(values_3d)
        if (allocated(values_2d_avg)) deallocate(values_2d_avg)
        if (allocated(values_2d_tmp)) deallocate(values_2d_tmp)
        if (allocated(values_3d_avg)) deallocate(values_3d_avg)
        if (allocated(values_3d_tmp)) deallocate(values_3d_tmp)
        if (write_spread_ncio) then
           deallocate(values_2d_sprd, values_3d_sprd)
        endif
        if (mype == 0) then
           call close_dataset(dseto)
           t2 = mpi_wtime()
           print *,'time to write ens mean on root',t2-t1
           write(6,'(3a,i5)')'Write ncio ensemble mean ',trim(filenameout),' iret = ', iret
           if (write_spread_ncio) then
             call close_dataset(dseto_sprd)
              write(6,'(3a,i5)')'Write ncio ensemble spread ',trim(filenameoutsprd),' iret = ', iret
           endif
        endif
!     endif

!    Deallocate structures and arrays
     call close_dataset(dset)

! Jump here if more mpi processors than files to process
  else
     write(6,'(a,i5)') 'No files to process for mpi task = ',mype
  endif

100 continue
  call mpi_barrier(mpi_comm_world,iret)

  if ( mype == 0 ) call w3tage('GETSIGENSMEAN_SMOOTH')

 deallocate(new_group_members)

 call mpi_finalize(iret)


end program getgribncensmeanp
