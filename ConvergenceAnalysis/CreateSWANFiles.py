import os

# The various resolutions and step sizes needed
dxinp = [0.00015, 0.00031, 0.00062, 0.00124, 0.00247, 0.00494, 0.00989]
dyinp = [0.00011, 0.00023, 0.00045, 0.00090, 0.00180, 0.00360, 0.00720]
mxinp = [640, 320, 160, 80, 40, 20, 10]
res = [12.5, 25, 50, 100, 200, 400, 800]

swn_file = """$-------- Section 1. Start --------------------------------------
$
PROJ 'BaskCoast' 'Run1'
$
MODE NONSTAT TWOD
$
$-------- Section 2. General ------------------------------------
$
SET LEVEL=2.25 NOR=90. NAUT
COORD SPHERICAL
$
$-------- Section 3. Model Grid ----------------------------------
$
$ Grid from external file (possibly variable)
CGRID CURVILINEAR MXC={} MYC={} EXCEPTION -99.0 &
  SECTOR DIR1=215. DIR2=55. MDC=80 FLOW=0.05 FHIGH=0.25 MSC=25
READGRID COORDINATES FAC=1. 'inp\GridSWAN_ZoomBtz_{}m.dat' IDLA=3 NHEDF=0 FREE
$
$ Bathymetry
INPGRID BOTTOM REGULAR XPINP=-1.617 YPINP=43.46 ALPINP=0. MXINP={} MYINP={} & 
  DXINP={} DYINP={}
READINP BOTTOM FAC=1. 'inp\BathySWAN_ZoomBtz_{}m.dat' IDLA=3 NHEDF=0 FREE
$
$-------- Section 4. Forcings ------------------------------------
$
$ Wind
$ No wind
$
$ Water level
$ If constant, to be imposed in "SET" command line
$
$-------- Section 5. Boundary/Initial Conditions ----------------
$
BOUNDNEST1 NEST 'out/BaskCoast_nesting.spec' OPEN
$
$-------- Section 6. Physics -------------------------------------
$
OFF WINDGROWTH
OFF QUADRUPL
OFF WCAPPING
$
FRICTION MADSEN KN=0.085
$
$ OFF BREAKING
BREAKING CONSTANT ALPHA=1.0 GAMMA=0.73
$
$-------- Section 7. Numerics ------------------------------------
$
PROP BSBT
$
$ - Non stationary mode -
NUM STOPC DABS=0.005 DREL=0.01 CURVAT=0.005 NPNTS=98.0 NONSTAT MXITNS=50
$
$-------- Section 9. Outputs --------------------------------------
$
OUTPUT OPTIONS '%' BLOCK NDEC=6
$
$ - Gridded outputs -
$
$ For post-processing with Matlab
BLOCK 'COMPGRID' NOHEAD 'out\BaskCoast_ZoomBtz_{}m.mat' LAY IDLA=3 &
      HSIGN TM02 DIR OUTPUT TBEGBLK=20181215.000000 DELTBLK=1 HR
$
BLOCK 'COMPGRID' NOHEADER 'BaskCoast_ZoomBtz_{}m_XP.dat' LAY IDLA=3 XP
BLOCK 'COMPGRID' NOHEADER 'BaskCoast_ZoomBtz_{}m_YP.dat' LAY IDLA=3 YP
BLOCK 'COMPGRID' NOHEADER 'BaskCoast_ZoomBtz_{}m_BOTLEV.dat' LAY IDLA=3 BOTLEV
$
$-------- Section 11. Run ----------------------------------------
$
$ - Stationary mode -
COMPUTE STAT TIME=20180701.000000
$
STOP
$"""


for i in range(len(dxinp)):
    if i != 0:
        continue
    mx = mxinp[i]
    dx = dxinp[i]
    dy = dyinp[i]
    r = res[i]

    swn = swn_file.format(mx, mx, r, mx, mx, dx, dy, r, r, r, r, r)

    with open("BaskCoast_ZoomBtz.swn", "w") as f:
        f.write(swn)

    cmd = "swanrun -input BaskCoast_ZoomBtz -omp 6"
    os.system(cmd)
