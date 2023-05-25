# Visco-elastic compressible formulation: julia -O3 --check-bounds=no Stokes2D_vep_egc.jl
USE_GPU  = false      # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
GPU_ID   = 0
do_viz   = true
do_gif   = false
do_save  = false
do_break = false
restart  = 0
condom   = true
GlobalVariable = true
runame   = "Stokes2Dvep_egc_crust"
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
if do_viz
    using Plots
end
using MAT, Printf, Statistics, LinearAlgebra
###########################################
import ParallelStencil: INDICES
ix,  iy  = INDICES[1], INDICES[2]
ixi, iyi = :($ix+1), :($iy+1)
macro  d2_xa(A)  esc(:( $A[$ixi+1,$iy   ] - 2.0*$A[$ixi,$iy ] + $A[$ixi-1,$iy   ] )) end
macro  d2_ya(A)  esc(:( $A[$ix   ,$iyi+1] - 2.0*$A[$ix ,$iyi] + $A[$ix   ,$iyi-1] )) end

###########################################
@views function Stokes2D_vep( (;nres, rel, Vdmp, Wdmp, pldmp, scV, scPt, scW, scλ) )
    # Physics
    Lx, Ly    =  50e3, 30e3
    pconf     =  0*250e6
    ρg        = -26487.0
    η0        =  1e50
    Δt        =  1e11/4
    rad       =  2e3
    K         =  2e10
    G         =  1e10
    G_inc     =  1e10/4.0
    ε_bg      =  1e-15
    C         =  1.75e7
    C_inc     =  1.75e7/4.0
    ϕ         =  30.0*π/180.0
    ψ         =  0*10.0*π/180
    ϕ_inc     =  30.0*π/180.0
    ψ_inc     =  0*π/180.0
    ndis      =  3.3         # Power Law exponent
    Qdis      =  186.5e3     # Activation energy
    Adis      =  3.1623e-26
    t         =  0.0
    Rg        =  8.314
    h         =  1.0*(-2e8)  # softening
    η_inc     =  1e21
    # - regularisation
    η_vp      =  2.5e20      # Kelvin VP
    c_grad    =  0*1e13      # gradient
    coss      =  0.0         # cosserat switch
    G_coss    =  1e10        # cosserat
    l_coss    =  100.0       # cosserat
    # - characteristic
    μc        =  G*Δt
    Lc        =  56400.0/4.0
    tc        =  1.0/abs(ε_bg)
    Tc        =  100.0
    # - derived units
    τc        =  μc/tc
    Vc        =  Lc/tc
    mc        =  τc*Lc*tc^2   # Kilograms
    Jc        =  mc*Lc^2/tc^2 
    # - nondim
    Lx, Ly    =  Lx/Lc, Ly/Lc
    Δt        =  Δt/tc
    rad       =  rad/Lc
    K         =  K/τc
    G         =  G/τc 
    G_inc     =  G_inc/τc
    ε_bg      =  ε_bg*tc
    η0        =  η0/μc
    C         =  C/τc
    C_inc     =  C_inc/τc
    η_vp      =  η_vp/μc
    c_grad    =  c_grad/Lc/Lc/τc
    h         =  h/τc
    G_coss    =  G_coss/τc # cosserat
    l_coss    =  l_coss/Lc # cosserat
    pconf     =  pconf/τc
    ρg        =  ρg/(τc/Lc)
    Adis      =  Adis/(τc^(-ndis)*1.0/tc);
    Qdis      =  Qdis/Jc;
    Rg        =  Rg/(Jc/Tc)
    Tsurf     =  293/Tc;                     # Surface temperature
    gradT     = -15/1e3/(Tc/Lc);             # Temperature gradient
    η_inc     = η_inc/(τc*tc)
    # Numerics
    nx        =  Int64(nres*4*16 - 2) # -2 due to overlength of array nx+2
    ny        =  Int64(nres*3*16 - 2) # -2 due to overlength of array ny+2
    nt        =  500*4
    nout_viz  =  10
    iterMax   =  1e5
    nout      =  1000
    ntloc     =  2     # number of maxloc iters
    ε_nl      =  1e-5  # tested and debugged vs Matlab with ε=5e-11 ;-)
    G_smooth  =  false
    nsm       =  2      # number of smoothing steps
    # Derived numerics
    Δx, Δy    = Lx/nx, Ly/ny
    _Δx, _Δy  = 1.0/Δx, 1.0/Δy
    l_coss2   = l_coss^2 # cosserat
    # Initialize coordinates
    xmin, xmax =  0., Lx 
    ymin, ymax = -Ly, 0. 
    xc        = LinRange(xmin+Δx/2, xmax-Δx/2, nx  )
    xv        = LinRange(xmin     , xmax     , nx+1)
    yc        = LinRange(ymin+Δy/2, ymax-Δy/2, ny  )
    yv        = LinRange(ymin     , ymax     , ny+1)
    # Array Initialisation
    Pt1       = @zeros(nx  ,ny  )
    Pt0       = @zeros(nx  ,ny  )
    RPt       = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    Vxe       = @zeros(nx+1,ny+2)
    Vye       = @zeros(nx+2,ny+1)
    εiidis    = @zeros(nx  ,ny  )
    εxx       = @zeros(nx  ,ny  )
    εyy       = @zeros(nx  ,ny  )
    εzz       = @zeros(nx  ,ny  )
    εxy       = @zeros(nx  ,ny  )
    εxyv      = @zeros(nx+1,ny+1)
    εxx1      = @zeros(nx  ,ny  )
    εyy1      = @zeros(nx  ,ny  )
    εzz1      = @zeros(nx  ,ny  )
    εxy1      = @zeros(nx  ,ny  )
    εxyv1     = @zeros(nx+1,ny+1)
    τzz       = @zeros(nx  ,ny  )
    τxy       = @zeros(nx  ,ny  )
    τxyv      = @zeros(nx+1,ny+1)
    τxx0      = @zeros(nx  ,ny  )
    τyy0      = @zeros(nx  ,ny  )
    τzz0      = @zeros(nx  ,ny  )
    τxy0      = @zeros(nx  ,ny  )
    τxyv0     = @zeros(nx+1,ny+1)
    dQdτxx    = @zeros(nx  ,ny  )
    dQdτyy    = @zeros(nx  ,ny  )
    dQdτzz    = @zeros(nx  ,ny  )
    dQdτxy    = @zeros(nx  ,ny  )
    dQdRxy    = @zeros(nx  ,ny  ) # cosserat
    dQdMxz    = @zeros(nx  ,ny  ) # cosserat
    dQdMyz    = @zeros(nx  ,ny  ) # cosserat
    Kxz       = @zeros(nx  ,ny+1) # cosserat
    Kxz1      = @zeros(nx  ,ny+1) # cosserat
    Kyz       = @zeros(nx+1,ny  ) # cosserat
    Kyz1      = @zeros(nx+1,ny  ) # cosserat
    Mxzc      = @zeros(nx  ,ny  ) # cosserat
    Myzc      = @zeros(nx  ,ny  ) # cosserat
    Mxz       = @zeros(nx  ,ny+1) # cosserat
    Myz       = @zeros(nx+1,ny  ) # cosserat
    Mxze      = @zeros(nx+2,ny+1) # cosserat
    Myze      = @zeros(nx+1,ny+2) # cosserat
    Rxyc      = @zeros(nx  ,ny  ) # cosserat
    Rxyv      = @zeros(nx+1,ny+1) # cosserat
    Mxzc0     = @zeros(nx  ,ny  ) # cosserat
    Myzc0     = @zeros(nx  ,ny  ) # cosserat
    Mxz0      = @zeros(nx  ,ny+1) # cosserat
    Myz0      = @zeros(nx+1,ny  ) # cosserat
    Rxyv0     = @zeros(nx+1,ny+1) # cosserat
    Kxzc1     = @zeros(nx  ,ny  ) # cosserat
    Kyzc1     = @zeros(nx  ,ny  ) # cosserat
    Wxyv      = @zeros(nx+1,ny+1) # cosserat
    Wxyv1     = @zeros(nx+1,ny+1) # cosserat
    Wz        = @zeros(nx+1,ny+1) # cosserat
    dWzdτ     = @zeros(nx+1,ny+1) # cosserat
    σxyv      = @zeros(nx+1,ny+1) # cosserat
    σyxv      = @zeros(nx+1,ny+1) # cosserat
    RW        = @zeros(nx+1,ny+1) # cosserat
    dτWz      = @zeros(nx+1,ny+1) # cosserat
    τii       = 1e6/τc.*@ones(nx  ,ny  )
    εii       = @zeros(nx  ,ny  ) 
    εiitot    = @zeros(nx  ,ny  ) # for visu
    εxxtot    = @zeros(nx  ,ny  )
    εyytot    = @zeros(nx  ,ny  )
    εzztot    = @zeros(nx  ,ny  )
    εxytot    = @zeros(nx  ,ny  )
    Kxztot    = @zeros(nx  ,ny+1) # cosserat
    Kyztot    = @zeros(nx+1,ny  ) # cosserat
    εiip      = @zeros(nx  ,ny  )
    εiip0     = @zeros(nx  ,ny  )
    εiip_x    = @zeros(nx+2,ny  )
    εiip_y    = @zeros(nx  ,ny+2)
    Rx        = @zeros(nx-1,ny  )
    Ry        = @zeros(nx  ,ny-1)
    Rpl       = @zeros(nx  ,ny  )
    dλdτ      = @zeros(nx  ,ny  )
    dVxdτ     = @zeros(nx-1,ny  )
    dVydτ     = @zeros(nx  ,ny-1)
    dτVx      = @zeros(nx-1,ny  )
    dτVy      = @zeros(nx  ,ny-1)
    dτλ       = @zeros(nx  ,ny  )
    Fc        = @zeros(nx  ,ny  )
    FcP       = @zeros(nx  ,ny  ) # physics (non-relaxed)
    Plc       = @zeros(nx  ,ny  )
    λc        = @zeros(nx  ,ny  )
    λcP       = @zeros(nx  ,ny  ) # physics (non-relaxed)
    # Initialisation
    xc2       = repeat(xc, 1, length(yc))
    yc2       = repeat(yc, 1, length(xc))'
    T         = Data.Array(Tsurf .+ yc2.*gradT)
    Rog       = ρg.*@ones(nx,ny-1)
    Pt        = Data.Array(reverse(-cumsum(ρg.*ones(nx,ny).*Δy, dims=2), dims=2) .+ 0.5*ρg.*Δy)
    τxx       = .- 0.1.*pconf.*@ones(nx  ,ny  )
    τyy       =    0.1.*pconf.*@ones(nx  ,ny  )
    T         = Array(T)
    Bdis      = Adis.^(.-1.0./ndis) .* exp.(Qdis./Rg./T./ndis)
    Cdis      = (2.0.*Bdis).^(.-ndis)
    Bdis      = Data.Array(Bdis)
    Cdis      = Data.Array(Cdis)
    T         = Data.Array(T)
    ηc0       =     η0.*@ones(nx  ,ny  )
    ηc        =     η0.*@ones(nx  ,ny  )
    ηv        =     η0.*@ones(nx+1,ny+1)
    Bdis      = Array(Bdis)
    ηc        = Bdis.*abs(ε_bg).^(1.0./ndis-1)
    ηc        = Data.Array(ηc)
    Bdis      = Data.Array(Bdis)
    η_ec      =  G.*Δt.*@ones(nx  ,ny  )
    η_ev      =  G.*Δt.*@ones(nx+1,ny+1)
    η_vec     =        @ones(nx  ,ny  )
    η_vev     =        @ones(nx+1,ny+1)
    η_vev_coss=        @ones(nx+1,ny+1)
    η_vepc    =        @ones(nx  ,ny  )
    η2        =  @zeros(nx  ,ny  )
    σyy       =  @zeros(nx  ,ny  )
    σxx       =  @zeros(nx  ,ny  )
    η_vepv    =         @ones(nx+1,ny+1)
    Sin_ψ     = sin(ψ).*@ones(nx  ,ny  )
    Sin_ϕ     = sin(ϕ).*@ones(nx  ,ny  )
    Cos_ϕ     = cos(ϕ).*@ones(nx  ,ny  )
    Cc        =      C.*@ones(nx  ,ny  ) # softening
    Hc        =      h.*@ones(nx  ,ny  ) # softening
    η_ev_coss = G_coss.*Δt.*@ones(nx+1,ny+1) # cosserat
    λc0       = @zeros(nx  ,ny  )
    Cc0       = @zeros(nx  ,ny  ) # softening
    η_vepc2   = @zeros(nx  ,ny  ) # maxloc tmp
    η_vepv2   = @zeros(nx+1,ny+1) # maxloc tmp
    # Weights vertices
    wSW       = @ones(nx+1,ny+1)
    wSE       = @ones(nx+1,ny+1)
    wNW       = @ones(nx+1,ny+1)
    wNE       = @ones(nx+1,ny+1)
    # Dummy avg tables vertices
    AvSW      = @zeros(nx+1,ny+1)
    AvSE      = @zeros(nx+1,ny+1)
    AvNW      = @zeros(nx+1,ny+1)
    AvNE      = @zeros(nx+1,ny+1)
    # Prepare
    min_Δxy2  = min(Δx,Δy)^2
    max_nxy   = max(nx,ny)
    dampX     = (1.0-Vdmp/nx)
    dampY     = (1.0-Vdmp/ny)
    dampW     = (1.0-Wdmp/min(nx,ny)) # cosserat
    damppl    = (1.0-pldmp/min(nx,ny)) # gradient
    sin_ψ_inc = sin(ψ_inc)
    sin_ϕ_inc = sin(ϕ_inc)
    cos_ϕ_inc = cos(ϕ_inc)

    appname   = "$(runame)_$(nx)x$(ny)"; println("Launching $(appname)")
    if do_gif
        dname = "viz_$(appname)"; ENV["GKSwstype"]="nul"; if isdir("$dname")==false mkdir("$dname") end; loadpath = "./$dname/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
    end
    if do_save
        outdir = string(@__DIR__, "/out_$(appname)"); if isdir("$outdir")==false mkdir("$outdir") end; println("Output file directory: ./out_$(appname)")
    end
    if do_break
        breakdir = string(@__DIR__, "/breakpoints_$(appname)"); if isdir("$breakdir")==false mkdir("$breakdir") end;
    end
    # init
    @parallel weights!(wSW, wSE, wNW, wNE)

    # Draw weak seed
    Sin_ψ     = sin(ψ)*@ones(nx  ,ny  )
    Sin_ϕ     = sin(ϕ)*@ones(nx  ,ny  )
    Cos_ϕ     = cos(ϕ)*@ones(nx  ,ny  )
    println(minimum(ηc*τc*tc))
    println(maximum(ηc*τc*tc))

    @parallel initial_1!(T, η_ev, Vxe, Vye, Cc, Sin_ψ, Sin_ϕ, Cos_ϕ, xc, yc, xv, yv, Lx, Ly, rad, ε_bg, sin_ψ_inc, sin_ϕ_inc, cos_ϕ_inc, -rad)

    if G_smooth == true
        fact = 5.0 # reproduce the Matlab script needs 5.0
        η_ec2  = η_ec
        Sin_ψ2 = Sin_ψ
        Sin_ϕ2 = Sin_ϕ
        Cos_ϕ2 = Cos_ϕ
        for ism = 1:nsm
            @parallel smooth!(η_ec2, η_ec, fact)
            @parallel smooth!(Sin_ψ2, Sin_ψ, fact)
            @parallel smooth!(Sin_ϕ2, Sin_ϕ, fact)
            @parallel smooth!(Cos_ϕ2, Cos_ϕ, fact)
            @parallel (1:size(η_ec2, 2)) bc_x!(η_ec2)
            @parallel (1:size(η_ec2, 1)) bc_y!(η_ec2)
            @parallel (1:size(Sin_ψ2,2)) bc_x!(Sin_ψ2)
            @parallel (1:size(Sin_ψ2,1)) bc_y!(Sin_ψ2)
            @parallel (1:size(Sin_ϕ2,2)) bc_x!(Sin_ϕ2)
            @parallel (1:size(Sin_ϕ2,1)) bc_y!(Sin_ϕ2)
            @parallel (1:size(Cos_ϕ2,2)) bc_x!(Cos_ϕ2)
            @parallel (1:size(Cos_ϕ2,1)) bc_y!(Cos_ϕ2)
            η_ec, η_ec2   = η_ec2, η_ec
            Sin_ψ, Sin_ψ2 = Sin_ψ2, Sin_ψ
            Sin_ϕ, Sin_ϕ2 = Sin_ϕ2, Sin_ϕ
            Cos_ϕ, Cos_ϕ2 = Cos_ϕ2, Cos_ϕ
        end
    end
    @parallel c2v!(η_ev, η_ec, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
    @parallel v2c!(η_ec, η_ev)                                             # avg like Matlab
    @parallel c2v!(η_ev, η_ec, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE) # avg like Matlab
    @parallel initial_2!(η_vec, η_vev, η_vev_coss, ηc, ηv, η_ec, η_ev, η_ev_coss)

    # Restart
    it0 = 1
    if restart > 0
        file = matopen(string("$(breakdir)/Breakpoint", @sprintf("%04d",restart),   ".mat") );
        τxx  = Data.Array(read(file,"Txx"))
        τyy  = Data.Array(read(file,"Tyy"))
        τzz  = Data.Array(read(file,"Tzz"))
        τxyv = Data.Array(read(file,"Txyv"))
        Pt   = Data.Array(read(file,"Pt"))
        Vxe  = Data.Array(read(file,"Vxe"))
        Vye  = Data.Array(read(file,"Vye"))
        Cc   = Data.Array(read(file,"Cc"))
        εiip = Data.Array(read(file,"Eiip"))
        Mxzc = Data.Array(read(file,"Mxzc"))
        Myzc = Data.Array(read(file,"Myzc"))
        Mxz  = Data.Array(read(file,"Mxz"))
        Myz  = Data.Array(read(file,"Myz"))
        Rxyv = Data.Array(read(file,"Rxyv"))
        Wz   = Data.Array(read(file,"Wz"))
        it0 = restart + 1
    end

    tim_evo=[]; τii_max=[]; itc_evo=[]; iterT=0
    for it = it0:nt
        @parallel store0ld_it_1!(τxx0, τyy0, τzz0, τxy0, τxyv0, Pt0, εiip0, Cc0, τxx, τyy, τzz, τxyv, Pt, εiip, Cc)
        @parallel store0ld_it_2!(Mxzc0, Myzc0, Mxz0, Myz0, Rxyv0, Mxzc, Myzc, Mxz, Myz, Rxyv) # cosserat
        @parallel reset!(λc)
        err0=3*ε_nl; err=2*ε_nl; iter=1; err_evo1=[]; err_evo2=[]
        err_rate = err0/err
        while err > ε_nl && iter < iterMax
            global niter = iter
            @parallel store0ld!(λc0, λc)
            @parallel store0ld!(ηc0, ηc)
            @parallel (1:size(Vye,2)) bc_x!(Vye)
            @parallel (1:size(Vxe,1)) bc_y!(Vxe)
            @parallel compute_∇V!(∇V, Vxe, Vye, _Δx, _Δy)
            @parallel compute_ε!(εxx, εyy, εxyv, εzz, Vxe, Vye, ∇V, _Δx, _Δy)
            @parallel v2c!(εxy, εxyv)
            @parallel compute_K_W_new!(Kxz, Kyz, Wxyv, Wz, Vxe, Vye, _Δx, _Δy) # cosserat
            if iter==1 || mod(iter,1)==0
                if iter>1 
                    @parallel compute_η!(τii, εiidis, ηc, ηc0, Cdis, τxx, τyy, τzz, τxy, Mxzc, Myzc, Rxyc, ndis, coss, l_coss, rel) 
                    @parallel set_viscous_seed!(ηc, xc, yc, xv, yv, Lx, Ly, rad, η_inc)
                end
                @parallel c2v!(ηv, ηc, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
                @parallel initial_2!(η_vec, η_vev, η_vev_coss, ηc, ηv, η_ec, η_ev, η_ev_coss)
                @parallel compute_τ!(εxx1, εyy1, εzz1, εxy1, εxyv1, τxx, τyy, τzz, τxy, εxx, εyy, εzz, εxy, εxyv, τxx0, τyy0, τzz0, τxy0, τxyv0, η_vec, η_ec, η_ev)
                @parallel compute_εiip!(εiip_x, εiip_y, εiip0, λc, Δt)
                @parallel (1:size(εiip_x,2)) bc_x!(εiip_x)
                @parallel (1:size(εiip_y,1)) bc_y!(εiip_y)
                @parallel compute_K_W1_new!( Wxyv1, Kxz1, Kyz1, Kxzc1, Kyzc1, Wxyv, Wz, Rxyv0, Kxz, Kyz, Mxz0, Myz0, Mxzc0, Myzc0, η_vev_coss, η_ev_coss, η_ev, η_ec, l_coss2) # cosserat
                @parallel compute_M_R_new!( Rxyv, Rxyc, Mxzc, Myzc, Kxzc1, Kyzc1, Wxyv1, η_vec, η_vev, l_coss2 ) # cosserat
                @parallel check_yield_1_new!(τii, Fc, τxx, τyy, τzz, τxy, Pt, Sin_ϕ, Cos_ϕ, εiip_x, εiip_y, Cc0, Mxzc, Myzc, Rxyc, c_grad, l_coss, _Δx, _Δy, coss)
                if (mod(iter, nout)==0)  max_Fc = maximum(Fc);  @printf("Init  Fc = %1.3e \n", max_Fc*τc)  end
                @parallel compute_Pl_λ_dQ_new!(Plc, λcP, λc, Fc, dQdτxx, dQdτyy, dQdτzz, dQdτxy, dQdMxz, dQdMyz, dQdRxy, τii, τxx, τyy, τzz, τxy, Mxzc, Myzc, Rxyc, λc0, η_vec, Sin_ϕ, Sin_ψ, Cos_ϕ, Hc, η_vp, K, Δt, rel, l_coss, GlobalVariable)
                # Check with physical (non-relaxed) λ and Fc
                if (mod(iter, nout)==0 && GlobalVariable==false) || iter==1 # ACHTUNG: for REAL Physics yield check: Fc=FcP, λc=λcP
                    @parallel correct_new!(τxx, τyy, τzz, τxy, η_vec, Mxzc, Myzc, Rxyc, Pt1, εii, dQdτxx, dQdτyy, dQdτzz, dQdτxy, dQdMxz, dQdMyz, dQdRxy, λcP, εxx1, εyy1, εzz1, εxy1, Sin_ψ, Pt, Cc, Cc0, Hc, Kxzc1, Kyzc1, Wxyv1, l_coss, l_coss2, K, Δt, coss)
                    @parallel check_yield_2_new!(τii, FcP, τxx, τyy, τzz, τxy, Pt1, λcP, Sin_ϕ, Cos_ϕ, εiip_x, εiip_y, Cc, Mxzc, Myzc, Rxyc, l_coss, η_vp, c_grad, _Δx, _Δy, coss) # cosserat
                    max_FcP = maximum(FcP);  @printf("Check Fc_phys = %1.3e \n", max_FcP*τc)
                end
                # Compute with relaxed λ and Fc
                @parallel correct_new!(τxx, τyy, τzz, τxy, η_vec, Mxzc, Myzc, Rxyc, Pt1, εii, dQdτxx, dQdτyy, dQdτzz, dQdτxy, dQdMxz, dQdMyz, dQdRxy, λc, εxx1, εyy1, εzz1, εxy1, Sin_ψ, Pt, Cc, Cc0, Hc, Kxzc1, Kyzc1, Wxyv1, l_coss, l_coss2, K, Δt, coss)
                @parallel check_yield_2_new!(τii, Fc, τxx, τyy, τzz, τxy, Pt1, λc, Sin_ϕ, Cos_ϕ, εiip_x, εiip_y, Cc, Mxzc, Myzc, Rxyc, l_coss, η_vp, c_grad, _Δx, _Δy, coss) # cosserat
                if (mod(iter, nout)==0) || iter==1 max_Fc = maximum(Fc);  @printf("Check Fc_rel  = %1.3e \n", max_Fc*τc)  end
                @parallel compute_η_vep!(η_vepc, ηc0, τii, εii, rel)
                @parallel c2v!(η_vepv, η_vepc, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
            end
            @parallel compute_τxyv_new!(τxyv, Rxyv, εxyv1, Wxyv1, η_vepv, coss)
            @parallel compute_Sxy_M!(σxyv, σyxv, Mxz, Myz, Mxze, Myze, τxyv, Rxyv, η_vepv, Kxz1, Kyz1, l_coss2, coss) # cosserat
            @parallel compute_σ!(σxx, σyy, Pt1, τxx, τyy)
            @parallel fs_bc!(σyy, σxyv, σyxv, η2, η_vepc)
            @parallel compute_Res!(RPt, Rx, Ry, RW, Rpl, Plc, Fc, Pt, Pt0, ∇V, Rog, σxx, σyy, σxyv, σyxv, Mxze, Myze, Δt, K, _Δx, _Δy, coss) # cosserat
            @parallel (1:size(RW,2)) bc_x0!(RW) # cosserat
            @parallel (1:size(RW,1)) bc_y0!(RW) # cosserat
            if mod(iter, nout)==0 || iter==1
                global norm_Rx, norm_Ry, norm_RPt, norm_RW, norm_Rpl
                norm_Rx  = norm(Rx)/length(Rx)
                norm_Ry  = norm(Ry)/length(Ry)
                norm_RPt = norm(RPt)/length(RPt)
                norm_RW  = norm(RW)/length(RW)
                nplc = sum(Plc.==1.0)
                norm_Rpl = norm(Rpl)/nplc
                if iter>1 err0 = err end
                err = maximum([norm_Rx, norm_Ry, norm_RPt, norm_RW]);
                push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_RPt, norm_RW])); push!(err_evo2, iter)
                @printf("> it %d, iter %d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, RPt=%1.3e, RW=%1.3e, Rpl=%1.3e] --- rel = %2.4e \n", it, niter, err, norm_Rx, norm_Ry, norm_RPt, norm_RW, norm_Rpl, rel)
            	if iter>1
                    err_rate = err0/err
                    @printf("> err rate = %2.4lf\n", err_rate)
                end
                @printf("> dt = %2.2e, dtC = %2.2e, dtE = %2.2e\n", Δt*tc, Δx/maximum(Vxe)/4.1*tc, 1.0/maximum(εii) / (Lx/Δx) * tc )
                if norm_Rx>1e3 || isnan(norm_Rx)
            		println("oups..."); file = matopen("J_Rx_$(it)_NaN$(iter).mat", "w"); write(file, "J_Rx", Array(Rx)); close(file)
            		error("too bad...")
            	end
            end
            # update
            for iloc = 1:ntloc
                @parallel compute_maxloc!(η_vepc2, η_vepv2, η_vepc, η_vepv)
                @parallel (1:size(η_vepc2,2)) bc_x!(η_vepc2)
                @parallel (1:size(η_vepc2,1)) bc_y!(η_vepc2)
                @parallel (1:size(η_vepv2,2)) bc_x!(η_vepv2)
                @parallel (1:size(η_vepv2,1)) bc_y!(η_vepv2)
                η_vepc, η_vepc2 = η_vepc2, η_vepc
                η_vepv, η_vepv2 = η_vepv2, η_vepv
            end
            @parallel compute_dτ!(dτVx, dτVy, dτPt, dτWz, dτλ, η_vepc, η_vepv, scPt, scV, scW, scλ, min_Δxy2, max_nxy) # cosserat
            @parallel compute_dV!(dVxdτ, dVydτ, dWzdτ, dλdτ, Rx, Ry, RW, Rpl, Plc, dampX, dampY, dampW, damppl) # cosserat
            @parallel update_VP!(Vxe, Vye, Pt, Wz, λc, dVxdτ, dVydτ, RPt, dWzdτ, dλdτ, dτVx, dτVy, dτPt, dτWz, dτλ)
            @parallel Vy_bc!(Vye, Vxe, Pt, η2, η_vepc, _Δx, _Δy)
            @parallel Vx_bc!(Vxe)

            iter+=1; iterT+=1
        end
        # Include plastic correction in converged pressure, εiip, softening
        @parallel update_Pt_εiip_Cc!(Pt, εiip, Cc, Hc, Pt1, εiip0, λc, Cc0, Δt, C)
        @parallel update_evo1!(εxxtot, εyytot, εzztot, εxytot, Kxztot, Kyztot, εxx, εyy, εzz, εxy, Kxz, Kyz, ∇V, Δt)
        @parallel update_evo2!(εiitot, εxxtot, εyytot, εzztot, εxytot, Kxztot, Kyztot, εii, εxx, εyy, εzz, εxy, Kxz, Kyz, l_coss, coss)
        t = t + Δt
        if do_viz
            # record loading
            push!(tim_evo, t); push!(τii_max, mean(Array(τii))*τc); push!(itc_evo, niter)
            default(size=(600,300))
            p1 = heatmap(xc*Lc,yc*Lc, log10.(Array(εii*1.0/tc))', aspect_ratio=1, xlims=(minimum(xv*Lc), maximum(xv)*Lc), ylims=(minimum(yv)*Lc, maximum(yv)*Lc), c=cgrad(:roma, rev = true), title="log10 εii")
            p2 = heatmap(xc*Lc,yc*Lc, Array(Cc*τc/1e6)', aspect_ratio=1, xlims=(minimum(xv*Lc), maximum(xv)*Lc), ylims=(minimum(yv)*Lc, maximum(yv)*Lc), c=cgrad(:roma, rev = true), title="C")
            p4 = heatmap(xv*Lc,yv*Lc,        Array(τii*τc/1e6)', aspect_ratio=1, xlims=(minimum(xv)*Lc, maximum(xv)*Lc), ylims=(minimum(yv)*Lc, maximum(yv)*Lc), c=cgrad(:roma, rev = true), title="τii" )#, clims=((0.0, 300))
            p3 = heatmap(xc*Lc,yc*Lc, Array(Pt1*τc/1e6)', aspect_ratio=1, xlims=(minimum(xv)*Lc, maximum(xv)*Lc), ylims=(minimum(yv)*Lc, maximum(yv)*Lc), c=cgrad(:roma, rev = true), title="P", clims=((0.000, 800)) ) #clims=((0.0002, 0.00059))
            if do_viz && do_gif  plot(p1, p2, p3, p4, dpi=200, layout = (2, 2)); frame(anim)
            else         display(plot(p1, p2, p3, p4, dpi=200, layout = (2, 2))) end
        end
        if do_save && mod(it,nout_viz)==0
            file = matopen(string("$(outdir)/Output", @sprintf("%04d",it),   ".mat"), "w");
            write(file, "Pt",   Array(Pt*τc));
            write(file, "Vx",   Array(Vxe*τc));
            write(file, "Vy",   Array(Vye*τc)); 
            write(file, "Cc",   Array(Cc*τc));
            write(file, "Tii",  Array(τii*τc));
            write(file, "Eiit", Array(εiitot));
            write(file, "Eiip", Array(εiip));
            write(file, "Eii", Array(εii*1.0/tc));
            write(file, "etav", Array(ηc*τc*tc));
            write(file, "etavep", Array(η_vepc*τc*tc));
            write(file, "xc", Array(xc*Lc));
            write(file, "yc", Array(yc*Lc));
            #------
            write(file, "Rx", Array(Rx));
            write(file, "Ry", Array(Ry));
            write(file, "RW", Array(RW));
            write(file, "RPt", Array(RPt));
            close(file)
        end
        if do_break && mod(it,nout_viz)==0
            file = matopen(string("$(breakdir)/Breakpoint", @sprintf("%04d",it),   ".mat"), "w");
            write(file, "Txx",  Array(τxx));
            write(file, "Tyy",  Array(τyy));
            write(file, "Tzz",  Array(τzz));
            write(file, "Txyv", Array(τxyv));
            write(file, "Pt",   Array(Pt));
            write(file, "Vxe",  Array(Vxe));
            write(file, "Vye",  Array(Vye)); 
            write(file, "Cc",   Array(Cc));
            write(file, "Eiip", Array(εiip));
            write(file, "Mxzc", Array(Mxzc));
            write(file, "Myzc", Array(Myzc));
            write(file, "Mxz",  Array(Mxz));
            write(file, "Myz",  Array(Myz));
            write(file, "Rxyv", Array(Rxyv));
            write(file, "Wz",   Array(Wz))
            close(file)
        end
    end
    if do_gif  gif(anim, "$(appname).gif", fps = 5)  end
    return
end

###########################################

@parallel_indices (ix,iy) function weights!(wSW::Data.Array, wSE::Data.Array, wNW::Data.Array, wNE::Data.Array)
    nxA = size(wSW,1)
    nyA = size(wSW,2)
    if (ix==nxA && iy<=nyA)  wSW[nxA, iy ] = 2  end # wSW(end,:) = 2;
    if (ix<=nxA && iy==nyA)  wSW[ix , nyA] = 2  end # wSW(:,end) = 2;
    if (ix==nxA && iy==nyA)  wSW[nxA, nyA] = 4  end # wSW(end,end) = 4;
    if (ix==1   && iy<=nyA)  wSW[1  , iy ] = 0  end # wSW(1,:) = 0;
    if (ix<=nxA && iy==1  )  wSW[ix , 1  ] = 0  end # wSW(:,1) = 0;

    if (ix==1   && iy<=nyA)  wSE[1  , iy ] = 2  end # wSE(1,:) = 2;
    if (ix<=nxA && iy==nyA)  wSE[ix , nyA] = 2  end # wSE(:,end) = 2;
    if (ix==1   && iy==nyA)  wSE[1  , nyA] = 4  end # wSE(1,end) = 4;
    if (ix==nxA && iy<=nyA)  wSE[nxA, iy ] = 0  end # wSE(end,:) = 0;
    if (ix<=nxA && iy==1  )  wSE[ix , 1  ] = 0  end # wSE(:,1) = 0;

    if (ix==nxA && iy<=nyA)  wNW[nxA, iy ] = 2  end # wNW(end,:) = 2;
    if (ix<=nxA && iy==1  )  wNW[ix , 1  ] = 2  end # wNW(:,1) = 2;
    if (ix==nxA && iy==1  )  wNW[nxA, 1  ] = 4  end # wNW(end,1) = 4;
    if (ix==1   && iy<=nyA)  wNW[1  , iy ] = 0  end # wNW(1,:) = 0;
    if (ix<=nxA && iy==nyA)  wNW[ix , nyA] = 0  end # wNW(:,end) = 0;

    if (ix==1   && iy<=nyA)  wNE[1  , iy ] = 2  end # wNE(1,:) = 2;
    if (ix<=nxA && iy==1  )  wNE[ix , 1  ] = 2  end # wNE(:,1) = 2;
    if (ix==1   && iy==1  )  wNE[1  , 1  ] = 4  end # wNE(1,1) = 4;
    if (ix==nxA && iy<=nyA)  wNE[nxA, iy ] = 0  end # wNE(end,:) = 0;
    if (ix<=nxA && iy==nyA)  wNE[ix , nyA] = 0  end # wNE(:,end) = 0;
    return
end

@parallel_indices (ix,iy) function c2v!(Av::Data.Array, Ac::Data.Array, AvSW::Data.Array, AvSE::Data.Array, AvNW::Data.Array, AvNE::Data.Array, wSW::Data.Array, wSE::Data.Array, wNW::Data.Array, wNE::Data.Array)
    if (ix>=2 && ix<=size(AvSW,1)   && iy>=2 && iy<=size(AvSW,2)  )  AvSW[ix, iy] = Ac[ix-1,iy-1]  end
    if (ix>=1 && ix<=size(AvSE,1)-1 && iy>=2 && iy<=size(AvSE,2)  )  AvSE[ix, iy] = Ac[ix  ,iy-1]  end
    if (ix>=2 && ix<=size(AvNW,1)   && iy>=1 && iy<=size(AvNW,2)-1)  AvNW[ix, iy] = Ac[ix-1,iy  ]  end
    if (ix>=1 && ix<=size(AvNE,1)-1 && iy>=1 && iy<=size(AvNE,2)-1)  AvNE[ix, iy] = Ac[ix  ,iy  ]  end
    if (ix<=size(Av,1) && iy<=size(Av,2))  Av[ix, iy] = 0.25*(wSW[ix, iy]*AvSW[ix, iy] + wSE[ix, iy]*AvSE[ix, iy] + wNW[ix, iy]*AvNW[ix, iy] + wNE[ix, iy]*AvNE[ix, iy] )  end
    return
end

@parallel function v2c!(Ac::Data.Array, Av::Data.Array)
    @all(Ac) = @av(Av)
    return
end

@parallel function reset!(A::Data.Array)
    @all(A) = 0.0
    return
end

@parallel function set_vel!(A1::Data.Array, A2::Data.Array, B1::Data.Array, B2::Data.Array)
    @all(A1) = @all(B1)
    @all(A2) = @all(B2)
    return
end

@parallel function store0ld!(A0::Data.Array, A::Data.Array)
    @all(A0) = @all(A)
    return
end

@parallel function smooth!(A2::Data.Array, A::Data.Array, fact::Data.Number)
    @inn(A2) = @inn(A) + 1.0/4.1/fact*(@d2_xi(A) + @d2_yi(A))
    return
end

@parallel_indices (iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel_indices (iy) function bc_x0!(A::Data.Array)
    A[1  , iy] = 0.0
    A[end, iy] = 0.0
    return
end

@parallel_indices (ix) function bc_y0!(A::Data.Array)
    A[ix, 1  ] = 0.0
    A[ix, end] = 0.0
    return
end

@parallel_indices (ix,iy) function initial_1!(T::Data.Array, η_ev::Data.Array, Vxe::Data.Array, Vye::Data.Array, Cc::Data.Array, Sin_ψ::Data.Array, Sin_ϕ::Data.Array, Cos_ϕ::Data.Array, xc, yc, xv, yv, Lx::Data.Number, Ly::Data.Number, rad::Data.Number, ε_bg::Data.Number, sin_ψ_inc::Data.Number, sin_ϕ_inc::Data.Number, cos_ϕ_inc::Data.Number, zfric::Data.Number)
    if (ix<=size(xc, 1)  && iy<=size(yc, 1) ) radc2 = (xc[ix]-Lx*0.0)*(xc[ix]-Lx*0.0) + (yc[iy]+Ly/2.0)*(yc[iy]+Ly/2.0)  end
    if (ix<=size(Vxe,1)   && iy<=size(Vxe,2)-2)  Vxe[ix,iy+1] = -ε_bg*(xv[ix]-Lx*0.0)  end
    if (ix<=size(Vye,1)-2 && iy<=size(Vye,2)  )  Vye[ix+1,iy] =  ε_bg*(yv[iy]-Ly*0.0)  end
    return
end

@parallel function initial_2!(η_vec::Data.Array, η_vev::Data.Array, η_vev_coss::Data.Array, ηc::Data.Array, ηv::Data.Array, η_ec::Data.Array, η_ev::Data.Array, η_ev_coss::Data.Array)
    @all(η_vec)      = 1.0/(1.0/@all(ηc) + 1.0/@all(η_ec))
    @all(η_vev)      = 1.0/(1.0/@all(ηv) + 1.0/@all(η_ev))
    @all(η_vev_coss) = 1.0/(1.0/@all(ηv) + 1.0/@all(η_ev_coss))
    return
end

@parallel function store0ld_it_1!(τxx0::Data.Array, τyy0::Data.Array, τzz0::Data.Array, τxy0::Data.Array, τxyv0::Data.Array, Pt0::Data.Array, εiip0::Data.Array, Cc0::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxyv::Data.Array, Pt::Data.Array, εiip::Data.Array, Cc::Data.Array)
    @all(τxx0)  = @all(τxx)
    @all(τyy0)  = @all(τyy)
    @all(τzz0)  = @all(τzz)
    @all(τxy0)  = @av(τxyv)
    @all(τxyv0) = @all(τxyv)
    @all(Pt0)   = @all(Pt)
    @all(εiip0) = @all(εiip)
    @all(Cc0)   = @all(Cc)
    return
end

@parallel function store0ld_it_2!(Mxzc0::Data.Array, Myzc0::Data.Array, Mxz0::Data.Array, Myz0::Data.Array, Rxyv0::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Mxz::Data.Array, Myz::Data.Array, Rxyv::Data.Array)
    @all(Mxzc0) = @all(Mxzc)
    @all(Myzc0) = @all(Myzc)
    @all(Mxz0)  = @all(Mxz)
    @all(Myz0)  = @all(Myz)
    @all(Rxyv0) = @all(Rxyv)
    return
end

@parallel function compute_∇V!(∇V::Data.Array, Vxe::Data.Array, Vye::Data.Array, _Δx::Data.Number, _Δy::Data.Number)
    @all(∇V) = _Δx*@d_xi(Vxe) + _Δy*@d_yi(Vye)
    return
end

@parallel function compute_ε!(εxx::Data.Array, εyy::Data.Array, εxyv::Data.Array, εzz::Data.Array, Vxe::Data.Array, Vye::Data.Array, ∇V::Data.Array, _Δx::Data.Number, _Δy::Data.Number)
    @all(εxx)  = _Δx*@d_xi(Vxe) - 1.0/3.0*@all(∇V)
    @all(εyy)  = _Δy*@d_yi(Vye) - 1.0/3.0*@all(∇V)
    @all(εzz)  =                - 1.0/3.0*@all(∇V) # DEBUG: 1/3 gives wrong results on GPU :-/
    @all(εxyv) = 0.5*(_Δy*@d_ya(Vxe) + _Δx*@d_xa(Vye))
    return
end

@parallel function compute_K_W_new!(Kxz::Data.Array, Kyz::Data.Array, Wxyv::Data.Array, Wz::Data.Array, Vxe::Data.Array, Vye::Data.Array, _Δx::Data.Number, _Δy::Data.Number)
    @all(Kxz)  = _Δx*@d_xa(Wz)
    @all(Kyz)  = _Δy*@d_ya(Wz)
    @all(Wxyv) = 0.5*(_Δy*@d_ya(Vxe) - _Δx*@d_xa(Vye))
    return
end

@parallel function compute_τ!(εxx1::Data.Array, εyy1::Data.Array, εzz1::Data.Array, εxy1::Data.Array, εxyv1::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, εxyv::Data.Array, τxx0::Data.Array, τyy0::Data.Array, τzz0::Data.Array, τxy0::Data.Array, τxyv0::Data.Array, η_vec::Data.Array, η_ec::Data.Array, η_ev::Data.Array)
    # trial strain components
    @all(εxx1)  = @all(εxx)  + @all(τxx0)/2.0/@all(η_ec)
    @all(εyy1)  = @all(εyy)  + @all(τyy0)/2.0/@all(η_ec)
    @all(εzz1)  = @all(εzz)  + @all(τzz0)/2.0/@all(η_ec)
    @all(εxy1)  = @all(εxy)  + @all(τxy0)/2.0/@all(η_ec)
    @all(εxyv1) = @all(εxyv) + @all(τxyv0)/2.0/@all(η_ev)
    # Deviatoric stress tensor
    @all(τxx) = 2.0*@all(η_vec)*@all(εxx1)
    @all(τyy) = 2.0*@all(η_vec)*@all(εyy1)
    @all(τzz) = 2.0*@all(η_vec)*@all(εzz1)
    @all(τxy) = 2.0*@all(η_vec)*@all(εxy1)
    return
end

@parallel function compute_εiip!(εiip_x::Data.Array, εiip_y::Data.Array, εiip0::Data.Array, λc::Data.Array, Δt::Data.Number)
    @inn_x(εiip_x) = @all(εiip0) + Δt*sqrt(2.0/3.0)*@all(λc)
    @inn_y(εiip_y) = @all(εiip0) + Δt*sqrt(2.0/3.0)*@all(λc)
    return
end

@parallel function compute_K_W1_new!(  Wxyv1::Data.Array, Kxz1::Data.Array, Kyz1::Data.Array, Kxzc1::Data.Array, Kyzc1::Data.Array, Wxyv::Data.Array, Wz::Data.Array, Rxyv0::Data.Array, Kxz::Data.Array, Kyz::Data.Array, Mxz0::Data.Array, Myz0::Data.Array, Mxzc0::Data.Array, Myzc0::Data.Array, η_vev_coss::Data.Array, η_ev_coss::Data.Array, η_ev::Data.Array, η_ec::Data.Array, l_coss2::Data.Number)
    @all(Wxyv1) =   @all(Wxyv) +  @all(Wz) - @all(Rxyv0)/2.0/@all(η_ev_coss)
    @all(Kxz1)  =   @all(Kxz)  +  @all(Mxz0)/2.0/@av_xa(η_ev)/l_coss2
    @all(Kyz1)  =   @all(Kyz)  +  @all(Myz0)/2.0/@av_ya(η_ev)/l_coss2
    @all(Kxzc1) = @av_ya(Kxz)  + @all(Mxzc0)/2.0/  @all(η_ec)/l_coss2
    @all(Kyzc1) = @av_xa(Kyz)  + @all(Myzc0)/2.0/  @all(η_ec)/l_coss2
    return
end

@parallel function compute_M_R_new!(Rxyv::Data.Array, Rxyc::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Kxzc1::Data.Array, Kyzc1::Data.Array, Wxyv1::Data.Array, η_vec::Data.Array, η_vev::Data.Array, l_coss2::Data.Number)
    @all(Rxyv) = -2.0*@all(η_vev)*@all(Wxyv1)
    @all(Rxyc) = -2.0*@all(η_vec)*@av(Wxyv1)
    @all(Mxzc) = l_coss2*2.0*  @all(η_vec)*@all(Kxzc1)
    @all(Myzc) = l_coss2*2.0*  @all(η_vec)*@all(Kyzc1)
    return
end

@parallel function compute_η!(τii::Data.Array, εiidis::Data.Array, ηc::Data.Array, ηc0::Data.Array, Cdis::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Rxyc::Data.Array, ndis::Data.Number, coss::Data.Number, l_coss::Data.Number, rel::Data.Number)
    # Rheology
    @all(τii)    = sqrt( 0.5*(@all(τxx)*@all(τxx) + @all(τyy)*@all(τyy) + @all(τzz)*@all(τzz) + 2.0*@all(τxy)*@all(τxy) + coss*(@all(Mxzc)/l_coss)*(@all(Mxzc)/l_coss) + coss*(@all(Myzc)/l_coss)*(@all(Myzc)/l_coss) + 2.0*coss*(@all(Rxyc)*@all(Rxyc)) ) )
    @all(εiidis) = @all(Cdis) * @all(τii)^ndis 
    @all(ηc)     = rel*( 0.5*@all(τii) / @all(εiidis) ) + (1.0-rel)*@all(ηc0)
    return
end  

@parallel_indices (ix,iy) function set_viscous_seed!(ηc::Data.Array, xc, yc, xv, yv, Lx::Data.Number, Ly::Data.Number, rad::Data.Number, η_inc::Data.Number )
    if (ix<=size(xc, 1)  && iy<=size(yc, 1) ) radc2 = (xc[ix]-Lx*0.0)*(xc[ix]-Lx*0.0) + (yc[iy]+Ly/2.0)*(yc[iy]+Ly/2.0)  end
    if (ix<=size(ηc,1) && iy<=size(ηc,2)) if (radc2<rad*rad)  ηc[ix,iy] = η_inc  end  end
    return
end

@parallel function check_yield_1_new!(τii::Data.Array, Fc::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Pt::Data.Array, Sin_ϕ::Data.Array, Cos_ϕ::Data.Array, Eiip_x::Data.Array, Eiip_y::Data.Array, Cc0::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Rxyc::Data.Array, c_grad::Data.Number, l_coss::Data.Number, _dx::Data.Number, _dy::Data.Number, coss::Data.Number)
    # Rheology
    @all(τii) = sqrt( 0.5*(@all(τxx)*@all(τxx) + @all(τyy)*@all(τyy) + @all(τzz)*@all(τzz) + 2.0*@all(τxy)*@all(τxy) + coss*(@all(Mxzc)/l_coss)*(@all(Mxzc)/l_coss) + coss*(@all(Myzc)/l_coss)*(@all(Myzc)/l_coss) + 2.0*coss*(@all(Rxyc)*@all(Rxyc)) ) )
    # Check yield
    @all(Fc) = @all(τii) - @all(Pt)*@all(Sin_ϕ) - @all(Cc0)*@all(Cos_ϕ) + c_grad*(_dx*_dx*@d2_xa(Eiip_x) + _dy*_dy*@d2_ya(Eiip_y))
    return
end

@parallel_indices (ix,iy) function compute_Pl_λ_dQ_new!(Plc::Data.Array, λcP::Data.Array, λc::Data.Array, Fc::Data.Array, dQdτxx::Data.Array, dQdτyy::Data.Array, dQdτzz::Data.Array, dQdτxy::Data.Array, dQdMxz::Data.Array, dQdMyz::Data.Array, dQdRxy::Data.Array, τii::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Rxyc::Data.Array, λc0::Data.Array, η_vec::Data.Array, Sin_ϕ::Data.Array, Sin_ψ::Data.Array, Cos_ϕ::Data.Array, Hc::Data.Array, η_vp::Data.Number, K::Data.Number, Δt::Data.Number, rel::Data.Number, l_coss::Data.Number, GlobalVariable::Bool)
    # Plastic nodes
    if (ix<=size(Plc,1) && iy<=size(Plc,2))  Plc[ix,iy] = 0.0  end # reset plastic node flags
    if (ix<=size(Fc,1)  && iy<=size(Fc,2))   if (Fc[ix,iy]>0.0)  Plc[ix,iy] = 1.0  end  end
    # λ
    if GlobalVariable == 0
        if (ix<=size(λc,1) && iy<=size(λc,2))  λcP[ix,iy] =       Plc[ix,iy]*(Fc[ix,iy]/(η_vec[ix,iy] + η_vp + K*Δt*Sin_ϕ[ix,iy]*Sin_ψ[ix,iy] + Hc[ix,iy]*sqrt(2.0/3.0)*Cos_ϕ[ix,iy]*Δt))  end
        if (ix<=size(λc,1) && iy<=size(λc,2))  λc[ix,iy]  = rel*( Plc[ix,iy]*(Fc[ix,iy]/(η_vec[ix,iy] + η_vp + K*Δt*Sin_ϕ[ix,iy]*Sin_ψ[ix,iy] + Hc[ix,iy]*sqrt(2.0/3.0)*Cos_ϕ[ix,iy]*Δt)) ) + (1.0-rel)*λc0[ix,iy]  end
    else
        if (ix<=size(λc,1) && iy<=size(λc,2))  λc[ix,iy]  =  Plc[ix,iy]*λc[ix,iy] end
        if (ix<=size(λc,1) && iy<=size(λc,2))  λcP[ix,iy] =  λc[ix,iy]            end
    end
    # dQ
    if (ix<=size(dQdτxx,1) && iy<=size(dQdτxx,2))  dQdτxx[ix,iy] = 0.5/τii[ix,iy]*τxx[ix,iy]  end
    if (ix<=size(dQdτyy,1) && iy<=size(dQdτyy,2))  dQdτyy[ix,iy] = 0.5/τii[ix,iy]*τyy[ix,iy]  end
    if (ix<=size(dQdτzz,1) && iy<=size(dQdτzz,2))  dQdτzz[ix,iy] = 0.5/τii[ix,iy]*τzz[ix,iy]  end
    if (ix<=size(dQdτxy,1) && iy<=size(dQdτxy,2))  dQdτxy[ix,iy] = 1.0/τii[ix,iy]*τxy[ix,iy]  end
    if (ix<=size(dQdMxz,1) && iy<=size(dQdMxz,2))  dQdMxz[ix,iy] = 0.5/τii[ix,iy]*(Mxzc[ix,iy]/l_coss)  end # cosserat
    if (ix<=size(dQdMyz,1) && iy<=size(dQdMyz,2))  dQdMyz[ix,iy] = 0.5/τii[ix,iy]*(Myzc[ix,iy]/l_coss)  end # cosserat
    if (ix<=size(dQdRxy,1) && iy<=size(dQdRxy,2))  dQdRxy[ix,iy] = 1.0/τii[ix,iy]*Rxyc[ix,iy]  end
    return
end

@parallel function correct_new!(τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, η_vec::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Rxyc::Data.Array, Pt1::Data.Array, εii::Data.Array, dQdτxx::Data.Array, dQdτyy::Data.Array, dQdτzz::Data.Array, dQdτxy::Data.Array, dQdMxz::Data.Array, dQdMyz::Data.Array, dQdRxy::Data.Array, λc::Data.Array, εxx1::Data.Array, εyy1::Data.Array, εzz1::Data.Array, εxy1::Data.Array, Sin_ψ::Data.Array, Pt::Data.Array, Cc::Data.Array, Cc0::Data.Array, Hc::Data.Array, Kxzc1::Data.Array, Kyzc1::Data.Array, Wxyv1::Data.Array, l_coss::Data.Number, l_coss2::Data.Number, K::Data.Number, Δt::Data.Number, coss::Data.Number)
    @all(τxx)  = 2.0*@all(η_vec)*(@all(εxx1) -     @all(λc)*@all(dQdτxx))
    @all(τyy)  = 2.0*@all(η_vec)*(@all(εyy1) -     @all(λc)*@all(dQdτyy))
    @all(τzz)  = 2.0*@all(η_vec)*(@all(εzz1) -     @all(λc)*@all(dQdτzz))
    @all(τxy)  = 2.0*@all(η_vec)*(@all(εxy1) - 0.5*@all(λc)*@all(dQdτxy))
    @all(Mxzc) = l_coss2*2.0*@all(η_vec)*(@all(Kxzc1) - @all(λc)*@all(dQdMxz)/l_coss)
    @all(Myzc) = l_coss2*2.0*@all(η_vec)*(@all(Kyzc1) - @all(λc)*@all(dQdMyz)/l_coss)
    @all(Rxyc) = -2.0*@all(η_vec)*(@av(Wxyv1) + 0.5*@all(λc)*@all(dQdRxy))
    @all(Cc)   = @all(Cc0) + Δt*sqrt(2.0/3.0)*@all(λc)*@all(Hc)
    @all(Pt1)  = @all(Pt) + K*Δt*@all(λc)*@all(Sin_ψ)
    @all(εii)  = sqrt( 0.5*( @all(εxx1)*@all(εxx1) + @all(εyy1)*@all(εyy1) + @all(εzz1)*@all(εzz1) + coss*(@all(Kxzc1)*l_coss)*(@all(Kxzc1)*l_coss) + coss*(@all(Kyzc1)*l_coss)*(@all(Kyzc1)*l_coss) + 2.0*@all(εxy1)*@all(εxy1)+ 2.0*@av(Wxyv1)*@av(Wxyv1) ) )
    return
end

@parallel function check_yield_2_new!(τii::Data.Array, Fc::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Pt1::Data.Array, λc::Data.Array, Sin_ϕ::Data.Array, Cos_ϕ::Data.Array, εiip_x::Data.Array, εiip_y::Data.Array, Cc::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Rxyc::Data.Array, l_coss::Data.Number, η_vp::Data.Number, c_grad::Data.Number, _Δx::Data.Number, _Δy::Data.Number, coss::Data.Number)
    # Rheology
    @all(τii) = sqrt( 0.5*(@all(τxx)*@all(τxx) + @all(τyy)*@all(τyy) + @all(τzz)*@all(τzz) + 2.0*@all(τxy)*@all(τxy) + coss*(@all(Mxzc)/l_coss)*(@all(Mxzc)/l_coss) + coss*(@all(Myzc)/l_coss)*(@all(Myzc)/l_coss) + 2.0*coss*(@all(Rxyc)*@all(Rxyc)) ) )
    # Check yield
    @all(Fc)  = @all(τii) - @all(Pt1)*@all(Sin_ϕ) - @all(Cc)*@all(Cos_ϕ) - η_vp*@all(λc) + c_grad*(_Δx*_Δx*@d2_xa(εiip_x) + _Δy*_Δy*@d2_ya(εiip_y))
    return
end

@parallel function compute_η_vep!(η_vepc::Data.Array, ηc0::Data.Array, τii::Data.Array, εii::Data.Array, rel::Data.Number)
    @all(η_vepc) = @all(τii)/2.0/@all(εii)
    return
end

@parallel function compute_τxyv_new!(τxyv::Data.Array, Rxyv::Data.Array, εxyv1::Data.Array, Wxyv1::Data.Array, η_vepv::Data.Array, coss::Data.Number)
    @all(τxyv) =       2.0*@all(η_vepv)*@all(εxyv1)
    @all(Rxyv) = -2.0*coss*@all(η_vepv)*@all(Wxyv1)
    return
end

@parallel function compute_Sxy_M!(σxyv::Data.Array, σyxv::Data.Array, Mxz::Data.Array, Myz::Data.Array, Mxze::Data.Array, Myze::Data.Array, τxyv::Data.Array, Rxyv::Data.Array, η_vepv::Data.Array, Kxz1::Data.Array, Kyz1::Data.Array, l_coss2::Data.Number, coss::Data.Number)
    @all(σxyv)   = @all(τxyv) + coss*@all(Rxyv)
    @all(σyxv)   = @all(τxyv) - coss*@all(Rxyv)
    @all(Mxz)    = l_coss2*2.0*@av_xa(η_vepv)*@all(Kxz1)
    @all(Myz)    = l_coss2*2.0*@av_ya(η_vepv)*@all(Kyz1)
    @inn_x(Mxze) = l_coss2*2.0*@av_xa(η_vepv)*@all(Kxz1)
    @inn_y(Myze) = l_coss2*2.0*@av_ya(η_vepv)*@all(Kyz1)
    return
end

@parallel function compute_maxloc!(η_vepc2::Data.Array, η_vepv2::Data.Array, η_vepc::Data.Array, η_vepv::Data.Array)
    @inn(η_vepc2) = @maxloc(η_vepc)
    @inn(η_vepv2) = @maxloc(η_vepv)
    return
end

@parallel function compute_dτ!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, dτWz::Data.Array, dτλ::Data.Array, η_vepc::Data.Array, η_vepv::Data.Array, scPt::Data.Number, scV::Data.Number, scW::Data.Number, scλ::Data.Number, min_Δxy2::Data.Number, max_nxy::Int)
    @all(dτVx) = min_Δxy2/(@av_xa(η_vepc))/4.1/scV
    @all(dτVy) = min_Δxy2/(@av_ya(η_vepc))/4.1/scV
    @all(dτPt) = 4.1*@all(η_vepc)/max_nxy/scPt
    @all(dτWz) = min_Δxy2/@all(η_vepv)/4.1/scW
    @all(dτλ)  = min_Δxy2/(@all(η_vepc))/4.1/scλ # sqrt(min_Δxy2*@all(η_vepc))/4.1/scλ # min(dx,dy)./sqrt(1./etac) / 4.1 ;
    return
end

@parallel function compute_σ!(σxx::Data.Array, σyy::Data.Array, Pt1::Data.Array, τxx::Data.Array, τyy::Data.Array)
    @all(σxx) = @all(τxx) - @all(Pt1)
    @all(σyy) = @all(τyy) - @all(Pt1)
    return
end

@parallel_indices (ix,iy) function fs_bc!(σyy::Data.Array, σxyv::Data.Array, σyxv::Data.Array, η2::Data.Array, η_vepc::Data.Array)
    if (ix<=size(σyy,1) && iy==size(σyy,2))   σyy[ix, iy]  = 0.0 end
    if (ix<=size(σyy,1) && iy==size(σxyv,2))  σxyv[ix, iy] = -σxyv[ix, iy-1] end
    if (ix<=size(σyy,1) && iy==size(σyxv,2))  σyxv[ix, iy] = -σyxv[ix, iy-1] end
    if (ix<=size(η2,1)  && iy<=size(η2,2))    η2[ix, iy]   =  η_vepc[ix, iy] end
    return
end

@parallel_indices (ix,iy) function Vy_bc!(Vye::Data.Array, Vxe::Data.Array, Pt::Data.Array, η2::Data.Array, η_vepc::Data.Array, _Δx::Data.Number, _Δy::Data.Number)
    if (2<=ix<=size(Vye,1)-1 && iy==size(Vye,2))  Vye[ix, iy] = Vye[ix, iy-1] + 3.0/2.0 * (Pt[ix-1, iy-1] / (2.0*η2[ix-1, iy-1]) + 1.0/3.0 * (Vxe[ix, iy] .- Vxe[ix-1, iy])*_Δx)/_Δy end
    return
end

@parallel_indices (ix,iy) function Vx_bc!(Vxe::Data.Array)
    if (ix<=size(Vxe,1) && iy==size(Vxe,2))  Vxe[ix, iy] = 0.0 end
    return
end

# @parallel function compute_Res!(RPt::Data.Array, Rx::Data.Array, Ry::Data.Array, RW::Data.Array, Pt1::Data.Array, Pt::Data.Array, Pt0::Data.Array, ∇V::Data.Array, Rog::Data.Array, τxx::Data.Array, τyy::Data.Array, σxyv::Data.Array, σyxv::Data.Array, Mxze::Data.Array, Myze::Data.Array, Δt::Data.Number, K::Data.Number, _Δx::Data.Number, _Δy::Data.Number, coss::Data.Number)
@parallel function compute_Res!(RPt::Data.Array, Rx::Data.Array, Ry::Data.Array, RW::Data.Array, Rpl::Data.Array, Plc::Data.Array, Fc::Data.Array, Pt::Data.Array, Pt0::Data.Array, ∇V::Data.Array, Rog::Data.Array, σxx::Data.Array, σyy::Data.Array, σxyv::Data.Array, σyxv::Data.Array, Mxze::Data.Array, Myze::Data.Array, Δt::Data.Number, K::Data.Number, _Δx::Data.Number, _Δy::Data.Number, coss::Data.Number)
    @all(RPt) =  -@all(∇V) - 1.0/(K*Δt)*(@all(Pt) - @all(Pt0))
    @all(Rx)  =  _Δx*@d_xa(σxx) + _Δy*@d_yi(σyxv)
    @all(Ry)  =  _Δy*@d_ya(σyy) + _Δx*@d_xi(σxyv) + @all(Rog)
    @all(RW)  = (_Δx*@d_xa(Mxze) + _Δy*@d_ya(Myze) - (@all(σyxv) - @all(σxyv)))*coss
    @all(Rpl) =  @all(Plc) * @all(Fc)
    return
end

@parallel function compute_dV!(dVxdτ::Data.Array, dVydτ::Data.Array, dWzdτ::Data.Array, dλdτ::Data.Array, Rx::Data.Array, Ry::Data.Array, RW::Data.Array, Rpl::Data.Array, Plc::Data.Array, dampX::Data.Number, dampY::Data.Number, dampW::Data.Number, damppl::Data.Number)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    @all(dWzdτ) = dampW*@all(dWzdτ) + @all(RW)
    @all(dλdτ)  = @all(Plc) * (damppl*@all(dλdτ) + @all(Rpl))
    return
end

@parallel function update_VP!(Vxe::Data.Array, Vye::Data.Array, Pt::Data.Array, Wz::Data.Array, λc::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, RPt::Data.Array, dWzdτ::Data.Array, dλdτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, dτWz::Data.Array, dτλ::Data.Array)
    @inn(Vxe) = @inn(Vxe) + @all(dτVx)*@all(dVxdτ) # assumes Dirichlet E/W
    @inn(Vye) = @inn(Vye) + @all(dτVy)*@all(dVydτ) # assumes Dirichlet N/S
    @all(Pt)  = @all(Pt)  + @all(dτPt)*@all(RPt)
    @inn(Wz)  = @inn(Wz)  + @inn(dτWz)*@inn(dWzdτ)
    @all(λc)  = @all(λc)  + @all(dτλ) *@all(dλdτ)
    return
end

@parallel_indices (ix,iy) function update_Pt_εiip_Cc!(Pt::Data.Array, εiip::Data.Array, Cc::Data.Array, Hc::Data.Array, Pt1::Data.Array, εiip0::Data.Array, λc::Data.Array, Cc0::Data.Array, Δt::Data.Number, C::Data.Number)
    if (ix<=size(Pt,1)   && iy<=size(Pt,2)  )  Pt[ix,iy]   = Pt1[ix,iy]  end
    if (ix<=size(εiip,1) && iy<=size(εiip,2))  εiip[ix,iy] = εiip0[ix,iy] + Δt*sqrt(2.0/3.0)*λc[ix,iy]  end
    if (ix<=size(Cc,1)   && iy<=size(Cc,2)  )  Cc[ix,iy]   = Cc0[ix,iy]   + Δt*sqrt(2.0/3.0)*λc[ix,iy]*Hc[ix,iy]  end
    if (ix<=size(Hc,1)   && iy<=size(Hc,2)  )  if (Cc[ix,iy]<C/2.0)  Hc[ix,iy] = 0.0  end  end # Limit on softening
    return
end

@parallel function update_evo1!(εxxtot::Data.Array, εyytot::Data.Array, εzztot::Data.Array, εxytot::Data.Array, Kxztot::Data.Array, Kyztot::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, Kxz::Data.Array, Kyz::Data.Array, ∇V::Data.Array, Δt::Data.Number)
    @all(εxxtot) = @all(εxxtot) + Δt*(@all(εxx) + 1.0/3.0*@all(∇V))
    @all(εyytot) = @all(εyytot) + Δt*(@all(εyy) + 1.0/3.0*@all(∇V))
    @all(εzztot) = @all(εzztot) + Δt*(@all(εzz) + 1.0/3.0*@all(∇V))
    @all(εxytot) = @all(εxytot) + Δt* @all(εxy)
    @all(Kxztot) = @all(Kxztot) + Δt* @all(Kxz)
    @all(Kyztot) = @all(Kyztot) + Δt* @all(Kyz)
    return
end

@parallel function update_evo2!(εiitot::Data.Array, εxxtot::Data.Array, εyytot::Data.Array, εzztot::Data.Array, εxytot::Data.Array, Kxztot::Data.Array, Kyztot::Data.Array, εii::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, Kxz::Data.Array, Kyz::Data.Array, l_coss::Data.Number, coss::Data.Number)
    @all(εiitot) = sqrt( 0.5*( @all(εxxtot)*@all(εxxtot) + @all(εyytot)*@all(εyytot) + @all(εzztot)*@all(εzztot) + 2.0*@all(εxytot)*@all(εxytot) + coss*(@av_ya(Kxztot)*l_coss)*(@av_ya(Kxztot)*l_coss) + coss*(@av_xa(Kyztot)*l_coss)*(@av_xa(Kyztot)*l_coss) ) )
    @all(εii)    = sqrt( 0.5*( @all(εxx)   *@all(εxx)    + @all(εyy)   *@all(εyy)    + @all(εzz   )*@all(εzz)    + 2.0*@all(εxy)   *@all(εxy)    + coss*(@av_ya(Kxz)*l_coss)   *(@av_ya(Kxz)*l_coss)    + coss*(@av_xa(Kyz)*l_coss)   *(@av_xa(Kyz)*l_coss)    ) )
    return
end

@parallel function computeEii!(εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, Kxzc::Data.Array, Kyzc::Data.Array, Wxyv::Data.Array)
    @all(εii)  = sqrt( 0.5*( @all(εxx)*@all(εxx) + @all(εyy)*@all(εyy) + @all(εzz)*@all(εzz) + coss*(@all(Kxzc)*l_coss)*(@all(Kxzc)*l_coss) + coss*(@all(Kyzc)*l_coss)*(@all(Kyzc)*l_coss) + 2.0*@all(εxy)*@all(εxy)+ 2.0*@av(Wxyv)*@av(Wxyv) ) )
return
end

###########################################
nres      =  1::Int64         
Vdmp      =  2.0
Wdmp      =  3.0 # cosserat
pldmp     =  2.0
scV       =  4.0
scPt      =  4.1 
scW       =  0.25  # cosserat
scλ       =  0.25
rel       =  1e-3
Stokes2D_vep( (nres=nres, rel=rel, Vdmp=Vdmp, Wdmp=Wdmp, pldmp=pldmp, scV=scV, scPt=scPt, scW=scW, scλ=scλ) ) 

