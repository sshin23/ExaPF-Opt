using SuiteSparse, ExaPF,MadNLP, ExaOpt, CUDA, CUDAKernels, MadNLPGPU
using LinearAlgebra, BlockPowerFlow, KernelAbstractions
include(joinpath(dirname(pathof(ExaOpt)), "..", "test", "cusolver.jl"))

nbatches = 100
datafile = "../../data/case118.m"
aug = ExaOpt.instantiate_auglag_model(
    datafile;
    line_constraints=true, # throws error when true
    device=CUDADevice(),
    nbatches=nbatches
)

aug.ρ = 0.

@assert CUDA.has_cuda_gpu()
linear_solver = MadNLPLapackGPU
mnlp = ExaOpt.ExaNLPModel(aug)
options = Dict{Symbol, Any}(
    :tol=>1e-5, 
    :print_level=>MadNLP.DEBUG,
    :kkt_system=>MadNLP.DENSE_KKT_SYSTEM,
    :linear_solver=>linear_solver,
)

madopt = MadNLP.Options(linear_solver=linear_solver)
MadNLP.set_options!(madopt,options,Dict())
KKT = MadNLP.DenseKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
ipp = MadNLP.InteriorPointSolver{KKT}(mnlp, madopt; option_linear_solver=options)

@time MadNLP.optimize!(ipp)


