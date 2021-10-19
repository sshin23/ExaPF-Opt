using SuiteSparse, ExaPF,MadNLP, ExaOpt, CUDA, CUDAKernels, MadNLPGPU
using LinearAlgebra, BlockPowerFlow, KernelAbstractions
using MadNLP: DenseKKTSystem
include(joinpath(dirname(pathof(ExaOpt)), "..", "test", "cusolver.jl"))
include("condensedkktsystem.jl")


datafile = "../../data/case118.m"
nbatch_hessian = 100

# dev = CUDADevice()
# KKT = DenseCondensedKKTSystem{Float64, CuVector{Float64}, CuMatrix{Float64}}
# linear_solver = MadNLPLapackGPU

dev = CPU()
linear_solver = MadNLPLapackCPU
KKT = DenseCondensedKKTSystem{Float64, Vector{Float64}, Matrix{Float64}}


evaluator = ExaOpt.ReducedSpaceEvaluator(
    datafile;
    device=dev,
    nbatch_hessian = nbatch_hessian
)

nlp = ExaOpt.ExaNLPModel(evaluator)
madopt = MadNLP.Options(linear_solver=linear_solver, tol = 1e-6)
ipp = MadNLP.InteriorPointSolver{KKT}(nlp,madopt)

@time MadNLP.optimize!(ipp)
