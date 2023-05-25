# Visco-elastic compressible formulation: julia -O3 --check-bounds=no Stokes2D_vep_egc.jl
const USE_GPU = false      # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID  = 0
using Interpolations
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using MAT, Plots, Printf, Statistics, LinearAlgebra  # ATTENTION: plotting fails inside plotting library if using flag '--math-mode=fast'.
###########################################
import ParallelStencil: INDICES
ix,  iy  = INDICES[1], INDICES[2]
ixi, iyi = :($ix+1), :($iy+1)
macro  d2_xa(A)  esc(:( $A[$ixi+1,$iy   ] - 2.0*$A[$ixi,$iy ] + $A[$ixi-1,$iy   ] )) end
macro  d2_ya(A)  esc(:( $A[$ix   ,$iyi+1] - 2.0*$A[$ix ,$iyi] + $A[$ix   ,$iyi-1] )) end
###########################################
@views function Stokes2D_vep()
    do_viz    = true
    do_gif    = false
    do_save   = false
    runame    = "Stokes2Dvep_egc"
    # Physics
    Lx, Ly    =  14.1e3, 10.4e3
    pconf     =  250e6
    η0        =  1e50
    dt        =  8e8
    rad       =  5e2
    K         =  2e10
    G0        =  1e10
    ε_bg      =  2e-13
    C         =  1.75e7
    ϕ         =  30*π/180
    ψ         =   0*π/180
    ϕ_inc     =  30*π/180
    ψ_inc     =   0*π/180
    h         =  1.0*(-2e8)  # softening
    t         =  0.0
    # - regularisation
    η_vp      =  6.0*1e18    # Kelvin VP
    c_grad    =  0*5e12    # gradient
    coss      =  0.0         # cosserat switch
    G_coss    =  1e10        # cosserat
    l_coss    =  8e1         # cosserat
    # - characteristic
    μc        =  G0*dt
    Lc        =  Lx
    tc        =  1.0/ε_bg
    # - derived units
    τc        =  μc/tc
    Vc        =  Lc/tc
    # - nondim
    Lx, Ly    =  Lx/Lc, Ly/Lc
    dt        =  dt/tc
    rad       =  rad/Lc
    K         =  K/τc
    G0        =  G0/τc
    ε_bg      =  ε_bg*tc
    η0        =  η0/μc
    C         =  C/τc
    η_vp      =  η_vp/μc
    c_grad    =  c_grad/Lc/Lc/τc
    h         =  h/τc
    G_coss    =  G_coss/τc # cosserat
    l_coss    =  l_coss/Lc # cosserat
    pconf     =  pconf/τc
    # Numerics
    nx        =  4*3*16 - 2 # -2 due to overlength of array nx+2
    ny        =  4*2*16 - 2 # -2 due to overlength of array ny+2
    nt        =  5*60
    iterMax   =  3e5
    nout      =  500
    Vdmp      =  4.0
    Wdmp      =  4.0 # cosserat
    scV       =  1.0 #1.0
    scPt      =  4.0 #2.1      # now dτPt = 4.1*...      /1.0#/4.1
    scW       =  5e-3  # cosserat
    arel      = -9.1477e-04
    brel      = -0.5500
    rel       =  10^(brel)*10^(arel*nx);  #0.5   # 0.01 coss  #0.05 grad
    ntloc     =  4     # number of maxloc iters
    ε_nl      =  5e-8  # tested and debugged vs Matlab with ε=5e-11 ;-)
    G_smooth  =  false
    nsm       =  1      # number of smoothing steps
    # β_n       =  2.0 
    # Derived numerics
    dx, dy    = Lx/nx, Ly/ny
    _dx, _dy  = 1.0/dx, 1.0/dy
    l_coss2   = l_coss^2 # cosserat
    # Array Initialisation
    Pt1       = @zeros(nx  ,ny  )
    Pt0       = @zeros(nx  ,ny  )
    RPt       = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    ∇V        = @zeros(nx  ,ny  )
    Vxe       = @zeros(nx+1,ny+2)
    Vye       = @zeros(nx+2,ny+1)
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
    Rxyv      = @zeros(nx+1,ny+1) # cosserat
    Mxzc0     = @zeros(nx  ,ny  ) # cosserat
    Myzc0     = @zeros(nx  ,ny  ) # cosserat
    Mxz0      = @zeros(nx  ,ny+1) # cosserat
    Myz0      = @zeros(nx+1,ny  ) # cosserat
    Rxyv0     = @zeros(nx+1,ny+1) # cosserat
    Kxzc1     = @zeros(nx  ,ny  ) # cosserat
    Kyzc1     = @zeros(nx  ,ny  ) # cosserat
    Wxyv      = @zeros(nx+1,ny+1) # cosserat
    Wz        = @zeros(nx+1,ny+1) # cosserat
    dWzdτ     = @zeros(nx+1,ny+1) # cosserat
    Sxyv      = @zeros(nx+1,ny+1) # cosserat
    Syxv      = @zeros(nx+1,ny+1) # cosserat
    RW        = @zeros(nx+1,ny+1) # cosserat
    dτWz      = @zeros(nx+1,ny+1) # cosserat
    Tii       = @zeros(nx  ,ny  )
    Eii       = @zeros(nx  ,ny  )
    Eiitot    = @zeros(nx  ,ny  ) # for visu
    εxxtot    = @zeros(nx  ,ny  )
    εyytot    = @zeros(nx  ,ny  )
    εzztot    = @zeros(nx  ,ny  )
    εxytot    = @zeros(nx  ,ny  )
    Eiip      = @zeros(nx  ,ny  )
    Eiip0     = @zeros(nx  ,ny  )
    Eiip_x    = @zeros(nx+2,ny  )
    Eiip_y    = @zeros(nx  ,ny+2)
    Rx        = @zeros(nx-1,ny  )
    Ry        = @zeros(nx  ,ny-1)
    dVxdτ     = @zeros(nx-1,ny  )
    dVydτ     = @zeros(nx  ,ny-1)
    dτVx      = @zeros(nx-1,ny  )
    dτVy      = @zeros(nx  ,ny-1)
    Fc        = @zeros(nx  ,ny  )
    FcP       = @zeros(nx  ,ny  ) # physics (non-relaxed)
    Plc       = @zeros(nx  ,ny  )
    λc        = @zeros(nx  ,ny  )
    λcP       = @zeros(nx  ,ny  ) # physics (non-relaxed)
    Pt        =  pconf*@ones(nx  ,ny  )
    τxx       = -0.5*pconf*@ones(nx  ,ny  )
    τyy       =  0.5*pconf*@ones(nx  ,ny  )
    ηc        =     η0*@ones(nx  ,ny  )
    ηv        =     η0*@ones(nx+1,ny+1)
    η_ec      =  G0*dt*@ones(nx  ,ny  )
    η_ev      =  G0*dt*@ones(nx+1,ny+1)
    η_vec     =        @ones(nx  ,ny  )
    η_vev     =        @ones(nx+1,ny+1)
    η_vev_coss=        @ones(nx+1,ny+1)
    η_vepc    =        @ones(nx  ,ny  )
    η_vepv    =        @ones(nx+1,ny+1)
    Sin_ψ     = sin(ψ)*@ones(nx  ,ny  )
    Sin_ϕ     = sin(ϕ)*@ones(nx  ,ny  )
    Cos_ϕ     = cos(ϕ)*@ones(nx  ,ny  )
    Cc        =      C*@ones(nx  ,ny  ) # softening
    Hc        =      h*@ones(nx  ,ny  ) # softening
    η_ev_coss = G_coss*dt*@ones(nx+1,ny+1) # cosserat
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
    # Init coord
    xc        = LinRange(dx/2, Lx-dx/2, nx  )
    yc        = LinRange(dy/2, Ly-dy/2, ny  )
    xv        = LinRange(0.0 , Lx     , nx+1)
    yv        = LinRange(0.0 , Ly     , ny+1)
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = (1.0-Vdmp/nx)
    dampY     = (1.0-Vdmp/ny)
    dampW     = (1.0-Wdmp/min(nx,ny)) # cosserat
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
    # init
    @parallel weights!(wSW, wSE, wNW, wNE)
    @parallel initial_1!(η_ev, Vxe, Vye, Sin_ψ, Sin_ϕ, Cos_ϕ, xc, yc, xv, yv, Lx, Ly, rad, ε_bg, sin_ψ_inc, sin_ϕ_inc, cos_ϕ_inc)
    @parallel v2c!(η_ec, η_ev)

    # init random noise from HR MAT file
    file = matopen(string(@__DIR__, "/Noise_cohesion_1601_nstep500.mat")); noiseHR = read(file, "Cv"); close(file)
    (ncxHR, ncyHR) = size(noiseHR).-1
    xv_HR = LinRange(0.0, Lx, ncxHR+1)
    yv_HR = LinRange(0.0, Ly, ncyHR+1)
    itp1  = interpolate((xv_HR[:], yv_HR[:]), noiseHR, Gridded(Linear()))
    Cv    = Data.Array(itp1(xv, yv))
    @parallel v2c!(Cc, Cv)
    Cc    = Array(Cc)/τc
    Cmin  = mean(Cc)/10.0*0.0
    Cc    = Data.Array(Cc)
    kd = 1e-6; sm_t = 30.0
    dt_sm = dx^2/kd/4.1/5.0; nsm = ceil(sm_t/dt_sm); dt_sm = sm_t/nsm
    Cc2   = Cc
    for ism = 1:nsm
        @parallel  smooth!(Cc2, Cc, 1.0/(dt_sm*4.1/dx^2*kd) )
        @parallel (1:size(Cc2,2))  bc_x!(Cc2)
        @parallel (1:size(Cc2,1))  bc_y!(Cc2)
        Cc, Cc2 = Cc2, Cc
    end
    @printf("Diffusion of jump over a duration of %2.2f and using %02d steps\n", nsm*dt_sm, nsm)

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

    tim_evo=[]; τii_max=[]; itc_evo=[]; iterT=0
    for it = 1:nt
        @parallel store0ld_it_1!(τxx0, τyy0, τzz0, τxy0, τxyv0, Pt0, Eiip0, Cc0, τxx, τyy, τzz, τxyv, Pt, Eiip, Cc)
        @parallel store0ld_it_2!(Mxzc0, Myzc0, Mxz0, Myz0, Rxyv0, Mxzc, Myzc, Mxz, Myz, Rxyv) # cosserat
        @parallel reset!(λc)
        err=2*ε_nl; iter=1; err_evo1=[]; err_evo2=[]
        while err > ε_nl && iter < iterMax
            global niter = iter
            @parallel store0ld!(λc0, λc)
            @parallel (1:size(Vye,2)) bc_x!(Vye)
            @parallel (1:size(Vxe,1)) bc_y!(Vxe)
            @parallel compute_∇V!(∇V, Vxe, Vye, _dx, _dy)
            @parallel compute_ε!(εxx, εyy, εxyv, εzz, Vxe, Vye, ∇V, _dx, _dy)
            @parallel v2c!(εxy, εxyv)
            @parallel compute_K_W!(Kxz, Kyz, Wxyv, Wz, Vxe, Vye, _dx, _dy) # cosserat
            @parallel compute_τ!(εxx1, εyy1, εzz1, εxy1, εxyv1, τxx, τyy, τzz, τxy, εxx, εyy, εzz, εxy, εxyv, τxx0, τyy0, τzz0, τxy0, τxyv0, η_vec, η_ec, η_ev)
            @parallel compute_Eiip!(Eiip_x, Eiip_y, Eiip0, λc, dt)
            @parallel (1:size(Eiip_x,2)) bc_x!(Eiip_x)
            @parallel (1:size(Eiip_y,1)) bc_y!(Eiip_y)
            @parallel compute_Rxy_K!(Rxyv, Kxz1, Kyz1, Kxzc1, Kyzc1, Wxyv, Wz, Rxyv0, Kxz, Kyz, Mxz0, Myz0, Mxzc0, Myzc0, η_vev_coss, η_ev_coss, η_ev, η_ec, l_coss2) # cosserat
            @parallel compute_M!(Mxzc, Myzc, Kxzc1, Kyzc1, η_vec, l_coss2) # cosserat
            @parallel check_yield_1!(Tii, Fc, τxx, τyy, τzz, τxy, Pt, Sin_ϕ, Cos_ϕ, Eiip_x, Eiip_y, Cc0, Mxzc, Myzc, c_grad, l_coss, _dx, _dy, coss)
            if (mod(iter, nout)==0)  max_Fc = maximum(Fc);  @printf("Init  Fc = %1.3e \n", max_Fc)  end
            @parallel compute_Pl_λ_dQ!(Plc, λcP, λc, Fc, dQdτxx, dQdτyy, dQdτzz, dQdτxy, dQdMxz, dQdMyz, Tii, τxx, τyy, τzz, τxy, Mxzc, Myzc, λc0, η_vec, Sin_ϕ, Sin_ψ, Cos_ϕ, Hc, η_vp, K, dt, rel, l_coss)
            # Check with physical (non-relaxed) λ and Fc
            if (mod(iter, nout)==0) # ACHTUNG: for REAL Physics yield check: Fc=FcP, λc=λcP
                @parallel correct!(τxx, τyy, τzz, τxy, η_vec, Mxzc, Myzc, Pt1, Eii, dQdτxx, dQdτyy, dQdτzz, dQdτxy, dQdMxz, dQdMyz, λcP, εxx1, εyy1, εzz1, εxy1, Sin_ψ, Pt, Cc, Cc0, Hc, Kxzc1, Kyzc1, l_coss, l_coss2, K, dt, coss)
                @parallel check_yield_2!(Tii, FcP, τxx, τyy, τzz, τxy, Pt1, λcP, Sin_ϕ, Cos_ϕ, Eiip_x, Eiip_y, Cc, Mxzc, Myzc, l_coss, η_vp, c_grad, _dx, _dy, coss) # cosserat
                max_FcP = maximum(FcP);  @printf("Check Fc_phys = %1.3e \n", max_FcP)
            end
            # Compute with relaxed λ and Fc
            @parallel correct!(τxx, τyy, τzz, τxy, η_vec, Mxzc, Myzc, Pt1, Eii, dQdτxx, dQdτyy, dQdτzz, dQdτxy, dQdMxz, dQdMyz, λc, εxx1, εyy1, εzz1, εxy1, Sin_ψ, Pt, Cc, Cc0, Hc, Kxzc1, Kyzc1, l_coss, l_coss2, K, dt, coss)
            @parallel check_yield_2!(Tii, Fc, τxx, τyy, τzz, τxy, Pt1, λc, Sin_ϕ, Cos_ϕ, Eiip_x, Eiip_y, Cc, Mxzc, Myzc, l_coss, η_vp, c_grad, _dx, _dy, coss) # cosserat
            if (mod(iter, nout)==0)  max_Fc = maximum(Fc);  @printf("Check Fc_rel  = %1.3e \n", max_Fc)  end
            @parallel compute_η_vep!(η_vepc, Tii, Eii)
            @parallel c2v!(η_vepv, η_vepc, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
            @parallel compute_τxyv!(τxyv, εxyv1, η_vepv)
            @parallel compute_Sxy_M!(Sxyv, Syxv, Mxz, Myz, Mxze, Myze, τxyv, Rxyv, η_vepv, Kxz1, Kyz1, l_coss2, coss) # cosserat
            @parallel compute_Res!(RPt, Rx, Ry, RW, Pt1, Pt, Pt0, ∇V, τxx, τyy, Sxyv, Syxv, Mxze, Myze, dt, K, _dx, _dy, coss) # cosserat
            @parallel (1:size(RW,2)) bc_x0!(RW) # cosserat
            @parallel (1:size(RW,1)) bc_y0!(RW) # cosserat
            if mod(iter, nout)==0
                global norm_Rx, norm_Ry, norm_RPt, norm_RW
                norm_Rx  = norm(Rx)/length(Rx)
                norm_Ry  = norm(Ry)/length(Ry)
                norm_RPt = norm(RPt)/length(RPt)
                norm_RW  = norm(RW)/length(RW)
                err = maximum([norm_Rx, norm_Ry, norm_RPt, norm_RW]);
                push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_RPt, norm_RW])); push!(err_evo2, iter)
                @printf("> it %d, iter %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPt=%1.3e, norm_RW=%1.3e] \n\n", it, niter, err, norm_Rx, norm_Ry, norm_RPt, norm_RW)
            	if norm_Rx>1.0
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
            @parallel compute_dτ!(dτVx, dτVy, dτPt, dτWz, η_vepc, η_vepv, scPt, scV, scW, min_dxy2, max_nxy) # cosserat
            @parallel compute_dV!(dVxdτ, dVydτ, dWzdτ, Rx, Ry, RW, dampX, dampY, dampW) # cosserat
            @parallel update_VP!(Vxe, Vye, Pt, Wz, dVxdτ, dVydτ, RPt, dWzdτ, dτVx, dτVy, dτPt, dτWz)
            iter+=1; iterT+=1
        end
        # Include plastic correction in converged pressure, Eiip, softening
        @parallel update_Pt_Eiip_Cc!(Pt, Eiip, Cc, Hc, Pt1, Eiip0, λc, Cc0, dt, C)
        @parallel update_evo1!(εxxtot, εyytot, εzztot, εxytot, εxx, εyy, εzz, εxy, ∇V, dt)
        @parallel update_evo2!(Eiitot, εxxtot, εyytot, εzztot, εxytot, Kxz, Kyz, l_coss, coss)
        t = t + dt
        if do_viz
            # record loading
            push!(tim_evo, t); push!(τii_max, mean(Array(Tii))*τc); push!(itc_evo, niter)
            default(size=(600,300))
            p1 = plot(τii_max, legend=false, xlabel="# steps", ylabel="mean(Tii)", framestyle=:box, linewidth=2, markershape=:circle, markersize=3)
            p2 = heatmap(xc,yc, Array(Tii*τc)', aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(dy/2, Ly-dy/2), c=:hot, title="Tii")
            p3 = plot(itc_evo, legend=false, xlabel="# steps", ylabel="# PT iters", framestyle=:box, linewidth=2, markershape=:circle, markersize=3, title="(iter tot=$iterT)")
            p4 = heatmap(xc,yc, Array(Cc*τc)', aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(dy/2, Ly-dy/2), c=:hot, title="Cc") #clims=((0.0002, 0.00059))
            if do_viz && do_gif  plot(p1, p2, p3, p4, dpi=200); frame(anim)
            else         display(plot(p1, p2, p3, p4, dpi=200)) end
        end
        if do_save
            file = matopen(string("$(outdir)/Output", lpad(it,4,"0"),   ".mat"), "w");
            write(file, "J_Pt",   Array(Pt*τc));
            write(file, "J_Vx",   Array(Vxe*τc));
            write(file, "J_Vy",   Array(Vye*τc)); 
            write(file, "J_Cc",   Array(Cc*τc));
            write(file, "J_Tii",  Array(Tii*τc));
            write(file, "J_Eiit", Array(Eiitot));
            write(file, "J_Eiip", Array(Eiip));
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

@parallel_indices (ix,iy) function initial_1!(η_ev::Data.Array, Vxe::Data.Array, Vye::Data.Array, Sin_ψ::Data.Array, Sin_ϕ::Data.Array, Cos_ϕ::Data.Array, xc, yc, xv, yv, Lx::Data.Number, Ly::Data.Number, rad::Data.Number, ε_bg::Data.Number, sin_ψ_inc::Data.Number, sin_ϕ_inc::Data.Number, cos_ϕ_inc::Data.Number)
    if (ix<=size(Vxe,1)   && iy<=size(Vxe,2)-2)  Vxe[ix,iy+1] = -ε_bg*(xv[ix]-Lx*0.5)  end
    if (ix<=size(Vye,1)-2 && iy<=size(Vye,2)  )  Vye[ix+1,iy] =  ε_bg*(yv[iy]-Ly*1.0)  end
    return
end

@parallel function initial_2!(η_vec::Data.Array, η_vev::Data.Array, η_vev_coss::Data.Array, ηc::Data.Array, ηv::Data.Array, η_ec::Data.Array, η_ev::Data.Array, η_ev_coss::Data.Array)
    @all(η_vec) = 1.0/(1.0/@all(ηc) + 1.0/@all(η_ec))
    @all(η_vev) = 1.0/(1.0/@all(ηv) + 1.0/@all(η_ev))
    @all(η_vev_coss) = 1.0/(1.0/@all(ηv) + 1.0/@all(η_ev_coss))
    return
end

@parallel function store0ld_it_1!(τxx0::Data.Array, τyy0::Data.Array, τzz0::Data.Array, τxy0::Data.Array, τxyv0::Data.Array, Pt0::Data.Array, Eiip0::Data.Array, Cc0::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxyv::Data.Array, Pt::Data.Array, Eiip::Data.Array, Cc::Data.Array)
    @all(τxx0)  = @all(τxx)
    @all(τyy0)  = @all(τyy)
    @all(τzz0)  = @all(τzz)
    @all(τxy0)  = @av(τxyv)
    @all(τxyv0) = @all(τxyv)
    @all(Pt0)   = @all(Pt)
    @all(Eiip0) = @all(Eiip)
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

@parallel function compute_∇V!(∇V::Data.Array, Vxe::Data.Array, Vye::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(∇V) = _dx*@d_xi(Vxe) + _dy*@d_yi(Vye)
    return
end

@parallel function compute_ε!(εxx::Data.Array, εyy::Data.Array, εxyv::Data.Array, εzz::Data.Array, Vxe::Data.Array, Vye::Data.Array, ∇V::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(εxx)  = _dx*@d_xi(Vxe) - 1.0/3.0*@all(∇V)
    @all(εyy)  = _dy*@d_yi(Vye) - 1.0/3.0*@all(∇V)
    @all(εzz)  =                - 1.0/3.0*@all(∇V) # DEBUG: 1/3 gives wrong results on GPU :-/
    @all(εxyv) = 0.5*(_dy*@d_ya(Vxe) + _dx*@d_xa(Vye))
    return
end

@parallel function compute_K_W!(Kxz::Data.Array, Kyz::Data.Array, Wxyv::Data.Array, Wz::Data.Array, Vxe::Data.Array, Vye::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(Kxz)  = _dx*@d_xa(Wz)
    @all(Kyz)  = _dy*@d_ya(Wz)
    @all(Wxyv) = _dy*@d_ya(Vxe) - _dx*@d_xa(Vye)
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

@parallel function compute_Eiip!(Eiip_x::Data.Array, Eiip_y::Data.Array, Eiip0::Data.Array, λc::Data.Array, dt::Data.Number)
    @inn_x(Eiip_x) = @all(Eiip0) + dt*sqrt(2.0/3.0)*@all(λc)
    @inn_y(Eiip_y) = @all(Eiip0) + dt*sqrt(2.0/3.0)*@all(λc)
    return
end

@parallel function compute_Rxy_K!(Rxyv::Data.Array, Kxz1::Data.Array, Kyz1::Data.Array, Kxzc1::Data.Array, Kyzc1::Data.Array, Wxyv::Data.Array, Wz::Data.Array, Rxyv0::Data.Array, Kxz::Data.Array, Kyz::Data.Array, Mxz0::Data.Array, Myz0::Data.Array, Mxzc0::Data.Array, Myzc0::Data.Array, η_vev_coss::Data.Array, η_ev_coss::Data.Array, η_ev::Data.Array, η_ec::Data.Array, l_coss2::Data.Number)
    @all(Rxyv)  = -@all(η_vev_coss)*( (@all(Wxyv) + 2.0*@all(Wz)) - @all(Rxyv0)/@all(η_ev_coss) )
    @all(Kxz1)  =   @all(Kxz) +  @all(Mxz0)/2.0/@av_xa(η_ev)/l_coss2
    @all(Kyz1)  =   @all(Kyz) +  @all(Myz0)/2.0/@av_ya(η_ev)/l_coss2
    @all(Kxzc1) = @av_ya(Kxz) + @all(Mxzc0)/2.0/  @all(η_ec)/l_coss2
    @all(Kyzc1) = @av_xa(Kyz) + @all(Myzc0)/2.0/  @all(η_ec)/l_coss2
    return
end

@parallel function compute_M!(Mxzc::Data.Array, Myzc::Data.Array, Kxzc1::Data.Array, Kyzc1::Data.Array, η_vec::Data.Array, l_coss2::Data.Number)
    @all(Mxzc) = l_coss2*2.0*  @all(η_vec)*@all(Kxzc1)
    @all(Myzc) = l_coss2*2.0*  @all(η_vec)*@all(Kyzc1)
    return
end

@parallel function check_yield_1!(Tii::Data.Array, Fc::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Pt::Data.Array, Sin_ϕ::Data.Array, Cos_ϕ::Data.Array, Eiip_x::Data.Array, Eiip_y::Data.Array, Cc0::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, c_grad::Data.Number, l_coss::Data.Number, _dx::Data.Number, _dy::Data.Number, coss::Data.Number)
    # Rheology
    @all(Tii) = sqrt( 0.5*(@all(τxx)*@all(τxx) + @all(τyy)*@all(τyy) + @all(τzz)*@all(τzz) + 2.0*@all(τxy)*@all(τxy) + coss*(@all(Mxzc)/l_coss)*(@all(Mxzc)/l_coss) + coss*(@all(Myzc)/l_coss)*(@all(Myzc)/l_coss)) )
    # Check yield
    @all(Fc) = @all(Tii) - @all(Pt)*@all(Sin_ϕ) - @all(Cc0)*@all(Cos_ϕ) + c_grad*(_dx*_dx*@d2_xa(Eiip_x) + _dy*_dy*@d2_ya(Eiip_y))
    return
end

@parallel_indices (ix,iy) function compute_Pl_λ_dQ!(Plc::Data.Array, λcP::Data.Array, λc::Data.Array, Fc::Data.Array, dQdτxx::Data.Array, dQdτyy::Data.Array, dQdτzz::Data.Array, dQdτxy::Data.Array, dQdMxz::Data.Array, dQdMyz::Data.Array, Tii::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, λc0::Data.Array, η_vec::Data.Array, Sin_ϕ::Data.Array, Sin_ψ::Data.Array, Cos_ϕ::Data.Array, Hc::Data.Array, η_vp::Data.Number, K::Data.Number, dt::Data.Number, rel::Data.Number, l_coss::Data.Number)
    # Plastic nodes
    if (ix<=size(Plc,1) && iy<=size(Plc,2))  Plc[ix,iy] = 0.0  end # reset plastic node flags
    if (ix<=size(Fc,1)  && iy<=size(Fc,2))   if (Fc[ix,iy]>0.0)  Plc[ix,iy] = 1.0  end  end
    # λ
    if (ix<=size(λc,1) && iy<=size(λc,2))  λcP[ix,iy] =      Plc[ix,iy]*(Fc[ix,iy]/(η_vec[ix,iy] + η_vp + K*dt*Sin_ϕ[ix,iy]*Sin_ψ[ix,iy] + Hc[ix,iy]*sqrt(2.0/3.0)*Cos_ϕ[ix,iy]*dt))  end
    if (ix<=size(λc,1) && iy<=size(λc,2))  λc[ix,iy] = rel*( Plc[ix,iy]*(Fc[ix,iy]/(η_vec[ix,iy] + η_vp + K*dt*Sin_ϕ[ix,iy]*Sin_ψ[ix,iy] + Hc[ix,iy]*sqrt(2.0/3.0)*Cos_ϕ[ix,iy]*dt)) ) + (1.0-rel)*λc0[ix,iy]  end
    # dQ
    if (ix<=size(dQdτxx,1) && iy<=size(dQdτxx,2))  dQdτxx[ix,iy] = 0.5/Tii[ix,iy]*τxx[ix,iy]  end
    if (ix<=size(dQdτyy,1) && iy<=size(dQdτyy,2))  dQdτyy[ix,iy] = 0.5/Tii[ix,iy]*τyy[ix,iy]  end
    if (ix<=size(dQdτzz,1) && iy<=size(dQdτzz,2))  dQdτzz[ix,iy] = 0.5/Tii[ix,iy]*τzz[ix,iy]  end
    if (ix<=size(dQdτxy,1) && iy<=size(dQdτxy,2))  dQdτxy[ix,iy] = 1.0/Tii[ix,iy]*τxy[ix,iy]  end
    if (ix<=size(dQdMxz,1) && iy<=size(dQdMxz,2))  dQdMxz[ix,iy] = 0.5/Tii[ix,iy]*(Mxzc[ix,iy]/l_coss)  end # cosserat
    if (ix<=size(dQdMyz,1) && iy<=size(dQdMyz,2))  dQdMyz[ix,iy] = 0.5/Tii[ix,iy]*(Myzc[ix,iy]/l_coss)  end # cosserat
    return
end

@parallel function correct!(τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, η_vec::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, Pt1::Data.Array, Eii::Data.Array, dQdτxx::Data.Array, dQdτyy::Data.Array, dQdτzz::Data.Array, dQdτxy::Data.Array, dQdMxz::Data.Array, dQdMyz::Data.Array, λc::Data.Array, εxx1::Data.Array, εyy1::Data.Array, εzz1::Data.Array, εxy1::Data.Array, Sin_ψ::Data.Array, Pt::Data.Array, Cc::Data.Array, Cc0::Data.Array, Hc::Data.Array, Kxzc1::Data.Array, Kyzc1::Data.Array, l_coss::Data.Number, l_coss2::Data.Number, K::Data.Number, dt::Data.Number, coss::Data.Number)
    @all(τxx)  = 2.0*@all(η_vec)*(@all(εxx1) -     @all(λc)*@all(dQdτxx))
    @all(τyy)  = 2.0*@all(η_vec)*(@all(εyy1) -     @all(λc)*@all(dQdτyy))
    @all(τzz)  = 2.0*@all(η_vec)*(@all(εzz1) -     @all(λc)*@all(dQdτzz))
    @all(τxy)  = 2.0*@all(η_vec)*(@all(εxy1) - 0.5*@all(λc)*@all(dQdτxy))
    @all(Mxzc) = l_coss2*2.0*@all(η_vec)*(@all(Kxzc1) - @all(λc)*@all(dQdMxz)/l_coss)
    @all(Myzc) = l_coss2*2.0*@all(η_vec)*(@all(Kyzc1) - @all(λc)*@all(dQdMyz)/l_coss)
    @all(Cc)   = @all(Cc0) + dt*sqrt(2.0/3.0)*@all(λc)*@all(Hc)
    @all(Pt1)  = @all(Pt) + K*dt*@all(λc)*@all(Sin_ψ)
    @all(Eii)  = sqrt( 0.5*( @all(εxx1)*@all(εxx1) + @all(εyy1)*@all(εyy1) + @all(εzz1)*@all(εzz1) + 2.0*@all(εxy1)*@all(εxy1) + coss*(@all(Kxzc1)*l_coss)*(@all(Kxzc1)*l_coss) + coss*(@all(Kyzc1)*l_coss)*(@all(Kyzc1)*l_coss) ) )
    return
end
# ACHTUNG: for yield check: Fc=Fc_real, and λ=λ_phys !!!
@parallel function check_yield_2!(Tii::Data.Array, Fc::Data.Array, τxx::Data.Array, τyy::Data.Array, τzz::Data.Array, τxy::Data.Array, Pt1::Data.Array, λc::Data.Array, Sin_ϕ::Data.Array, Cos_ϕ::Data.Array, Eiip_x::Data.Array, Eiip_y::Data.Array, Cc::Data.Array, Mxzc::Data.Array, Myzc::Data.Array, l_coss::Data.Number, η_vp::Data.Number, c_grad::Data.Number, _dx::Data.Number, _dy::Data.Number, coss::Data.Number)
    # Rheology
    @all(Tii) = sqrt( 0.5*(@all(τxx)*@all(τxx) + @all(τyy)*@all(τyy) + @all(τzz)*@all(τzz) + 2.0*@all(τxy)*@all(τxy) + coss*(@all(Mxzc)/l_coss)*(@all(Mxzc)/l_coss) + coss*(@all(Myzc)/l_coss)*(@all(Myzc)/l_coss)) )
    # Check yield
    @all(Fc)  = @all(Tii) - @all(Pt1)*@all(Sin_ϕ) - @all(Cc)*@all(Cos_ϕ) - η_vp*@all(λc) + c_grad*(_dx*_dx*@d2_xa(Eiip_x) + _dy*_dy*@d2_ya(Eiip_y))
    return
end

@parallel function compute_η_vep!(η_vepc::Data.Array, Tii::Data.Array, Eii::Data.Array)
    @all(η_vepc) = @all(Tii)/2.0/@all(Eii)
    return
end

@parallel function compute_τxyv!(τxyv::Data.Array, εxyv1::Data.Array, η_vepv::Data.Array)
    @all(τxyv) = 2.0*@all(η_vepv)*@all(εxyv1)
    return
end

@parallel function compute_Sxy_M!(Sxyv::Data.Array, Syxv::Data.Array, Mxz::Data.Array, Myz::Data.Array, Mxze::Data.Array, Myze::Data.Array, τxyv::Data.Array, Rxyv::Data.Array, η_vepv::Data.Array, Kxz1::Data.Array, Kyz1::Data.Array, l_coss2::Data.Number, coss::Data.Number)
    @all(Sxyv) = @all(τxyv) + coss*@all(Rxyv)
    @all(Syxv) = @all(τxyv) - coss*@all(Rxyv)
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

@parallel function compute_dτ!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, dτWz::Data.Array, η_vepc::Data.Array, η_vepv::Data.Array, scPt::Data.Number, scV::Data.Number, scW::Data.Number, min_dxy2::Data.Number, max_nxy::Int)
    @all(dτVx) = min_dxy2/(@av_xa(η_vepc))/4.1/scV
    @all(dτVy) = min_dxy2/(@av_ya(η_vepc))/4.1/scV
    @all(dτPt) = 4.1*@all(η_vepc)/max_nxy/scPt
    @all(dτWz) = min_dxy2/@all(η_vepv)/4.1/scW
    return
end

@parallel function compute_Res!(RPt::Data.Array, Rx::Data.Array, Ry::Data.Array, RW::Data.Array, Pt1::Data.Array, Pt::Data.Array, Pt0::Data.Array, ∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, Sxyv::Data.Array, Syxv::Data.Array, Mxze::Data.Array, Myze::Data.Array, dt::Data.Number, K::Data.Number, _dx::Data.Number, _dy::Data.Number, coss::Data.Number)
    @all(RPt) =  -@all(∇V) - 1.0/(K*dt)*(@all(Pt) - @all(Pt0))
    @all(Rx)  =  _dx*@d_xa(τxx) + _dy*@d_yi(Syxv) - _dx*@d_xa(Pt1)
    @all(Ry)  =  _dy*@d_ya(τyy) + _dx*@d_xi(Sxyv) - _dy*@d_ya(Pt1)
    @all(RW)  = (_dx*@d_xa(Mxze) + _dy*@d_ya(Myze) - (@all(Syxv) - @all(Sxyv)))*coss
    return
end

@parallel function compute_dV!(dVxdτ::Data.Array, dVydτ::Data.Array, dWzdτ::Data.Array, Rx::Data.Array, Ry::Data.Array, RW::Data.Array, dampX::Data.Number, dampY::Data.Number, dampW::Data.Number)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    @all(dWzdτ) = dampW*@all(dWzdτ) + @all(RW)
    return
end

@parallel function update_VP!(Vxe::Data.Array, Vye::Data.Array, Pt::Data.Array, Wz::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, RPt::Data.Array, dWzdτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, dτWz::Data.Array)
    @inn(Vxe) = @inn(Vxe) + @all(dτVx)*@all(dVxdτ) # assumes Dirichlet E/W
    @inn(Vye) = @inn(Vye) + @all(dτVy)*@all(dVydτ) # assumes Dirichlet N/S
    @all(Pt)  = @all(Pt)  + @all(dτPt)*@all(RPt)
    @inn(Wz)  = @inn(Wz)  + @inn(dτWz)*@inn(dWzdτ)
    return
end

@parallel_indices (ix,iy) function update_Pt_Eiip_Cc!(Pt::Data.Array, Eiip::Data.Array, Cc::Data.Array, Hc::Data.Array, Pt1::Data.Array, Eiip0::Data.Array, λc::Data.Array, Cc0::Data.Array, dt::Data.Number, C::Data.Number)
    if (ix<=size(Pt,1)   && iy<=size(Pt,2)  )  Pt[ix,iy]   = Pt1[ix,iy]  end
    if (ix<=size(Eiip,1) && iy<=size(Eiip,2))  Eiip[ix,iy] = Eiip0[ix,iy] + dt*sqrt(2.0/3.0)*λc[ix,iy]  end
    if (ix<=size(Cc,1)   && iy<=size(Cc,2)  )  Cc[ix,iy]   = Cc0[ix,iy]   + dt*sqrt(2.0/3.0)*λc[ix,iy]*Hc[ix,iy]  end
    if (ix<=size(Hc,1)   && iy<=size(Hc,2)  )  if (Cc[ix,iy]<C/2.0)  Hc[ix,iy] = 0.0  end  end # Limit on softening
    return
end

@parallel function update_evo1!(εxxtot::Data.Array, εyytot::Data.Array, εzztot::Data.Array, εxytot::Data.Array, εxx::Data.Array, εyy::Data.Array, εzz::Data.Array, εxy::Data.Array, ∇V::Data.Array, dt::Data.Number)
    @all(εxxtot) = @all(εxxtot) + dt*(@all(εxx) + 1.0/3.0*@all(∇V))
    @all(εyytot) = @all(εyytot) + dt*(@all(εyy) + 1.0/3.0*@all(∇V))
    @all(εzztot) = @all(εzztot) + dt*(@all(εzz) + 1.0/3.0*@all(∇V))
    @all(εxytot) = @all(εxytot) + dt* @all(εxy)
    return
end

@parallel function update_evo2!(Eiitot::Data.Array, εxxtot::Data.Array, εyytot::Data.Array, εzztot::Data.Array, εxytot::Data.Array, Kxz::Data.Array, Kyz::Data.Array, l_coss::Data.Number, coss::Data.Number)
    @all(Eiitot) = sqrt( 0.5*( @all(εxxtot)*@all(εxxtot) + @all(εyytot)*@all(εyytot) + @all(εzztot)*@all(εzztot) + 2.0*@all(εxytot)*@all(εxytot) + coss*(@av_ya(Kxz)*l_coss)*(@av_ya(Kxz)*l_coss) + coss*(@av_xa(Kyz)*l_coss)*(@av_xa(Kyz)*l_coss) ) )
    return
end

Stokes2D_vep()
