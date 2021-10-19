import MadNLP: AbstractKKTSystem, AbstractNLPModel, get_nvar, get_ncon, nnz_jacobian, is_reduced, compress_jacobian!, compress_hessian!, get_raw_jacobian, mul!, jtprod!, set_jacobian_scaling!, build_kkt!, treat_fixed_variable!, eval_jac_wrapper!, eval_lag_hess_wrapper!, InteriorPointSolver, @trace, get_hessian, get_jacobian, get_minimize, jac_dense!, hess_dense!, diag!, is_valid, _build_dense_kkt_system!
    
struct DenseCondensedKKTSystem{T, VT, MT} <: AbstractKKTSystem{T, MT}
    n::Int
    m::Int
    ns::Int
    hess::MT
    jac::MT
    pr_diag::VT
    du_diag::VT
    diag_hess::VT
    # KKT system
    aug_com::MT
    aug_com_2::MT
    jac_work::MT
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    jacobian_scaling::VT
    # Buffers
    etc::Dict{Symbol, Any}
end

function DenseCondensedKKTSystem{T, VT, MT}(n, m, ind_ineq, ind_fixed) where {T, VT, MT}
    ns = length(ind_ineq)

    @assert ns == m
    
    hess = MT(undef, n, n)
    jac = MT(undef, m, n)
    pr_diag = VT(undef, n+m)
    du_diag = VT(undef, m)
    diag_hess = VT(undef, n)

    # If the the problem is unconstrained, then KKT system is directly equal
    # to the Hessian (+ some regularization terms)
    aug_com = if (m == 0)
        hess
    else
        MT(undef, n+ns+m, n+ns+m)
    end
    
    aug_com_2 = MT(undef, n, n)
    jac_work  = MT(undef, m, n)

    jacobian_scaling = VT(undef, m)

    # Init!
    fill!(aug_com, zero(T))
    fill!(aug_com_2, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))
    fill!(diag_hess, zero(T))
    fill!(jacobian_scaling, one(T))

    return DenseCondensedKKTSystem{T, VT, MT}(
        n,m,ns,
        hess, jac, pr_diag, du_diag, diag_hess, aug_com, aug_com_2, jac_work,
        ind_ineq, ind_fixed, jacobian_scaling, Dict{Symbol, Any}(),
    )
end

function DenseCondensedKKTSystem{T, VT, MT}(nlp::AbstractNLPModel, info_constraints=get_index_constraints(nlp)) where {T, VT, MT}
    return DenseCondensedKKTSystem{T, VT, MT}(
        get_nvar(nlp), get_ncon(nlp), info_constraints.ind_ineq, info_constraints.ind_fixed
    )
end

is_reduced(::DenseCondensedKKTSystem) = true

# Special getters for Jacobian
function get_jacobian(kkt::DenseCondensedKKTSystem)
    n = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    return view(kkt.jac, :, 1:n)
end
get_raw_jacobian(kkt::DenseCondensedKKTSystem) = kkt.jac

nnz_jacobian(kkt::DenseCondensedKKTSystem) = length(kkt.jac)
nnz_kkt(kkt::DenseCondensedKKTSystem) = length(kkt.aug_com)

function diag_add!(dest::AbstractMatrix, d1::AbstractVector, d2::AbstractVector)
    n = length(d1)
    @inbounds for i in 1:n
        dest[i, i] = d1[i] + d2[i]
    end
end

function _build_dense_kkt_system!(dest, hess, jac, pr_diag, du_diag, diag_hess, jacobian_scaling, n, m, ns)
    # Transfer Hessian
    for i in 1:n, j in 1:i
        if i == j
            dest[i, i] = pr_diag[i] + diag_hess[i]
        else
            dest[i, j] = hess[i, j]
            dest[j, i] = hess[j, i]
        end
    end
    # Transfer slack diagonal
    for i in 1:ns
        dest[i+n, i+n] = pr_diag[i+n]
    end
    # Transfer Jacobian
    for i in 1:m, j in 1:n 
        dest[i + n + ns, j] = jac[i, j]
        dest[j, i + n + ns] = jac[i, j]
    end
    # Transfer Jacobian
    for i in 1:ns
        dest[i + n + ns, i + n ] = -jacobian_scaling[i]
        dest[i + n, i + n + ns ] = -jacobian_scaling[i]
    end
    # Transfer dual regularization
    for i in 1:m
        dest[i + n + ns, i + n + ns] = du_diag[i]
    end
end

function build_kkt!(kkt::DenseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT}
    n = size(kkt.hess, 1)
    m = size(kkt.jac, 1)
    ns = length(kkt.ind_ineq)
    if m == 0 # If problem is unconstrained, just need to update the diagonal
        diag_add!(kkt.aug_com, kkt.diag_hess, kkt.pr_diag)
        diag_add!(kkt.aug_com_2, kkt.diag_hess, kkt.pr_diag)
    else # otherwise, we update the full matrix
        _build_dense_kkt_system!(
            kkt.aug_com, kkt.hess, kkt.jac, kkt.pr_diag, kkt.du_diag, kkt.diag_hess, kkt.jacobian_scaling,
            n, m, ns)
        # _build_dense_condensed_kkt_system!(kkt.aug_com, kkt.hess, kkt.jac, kkt.pr_diag, kkt.du_diag, kkt.diag_hess, n, m, ns)
    end
    treat_fixed_variable!(kkt)
end

function compress_jacobian!(kkt::DenseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT}
    kkt.jac .*= kkt.jacobian_scaling
    return
end

function compress_hessian!(kkt::DenseCondensedKKTSystem)
    # Transfer diagonal term for future regularization
    diag!(kkt.diag_hess, kkt.hess)
end

function mul!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    mul!(y, kkt.aug_com, x)
end

function jtprod!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    mul!(view(y,1:kkt.n), kkt.jac', x) 
    y[kkt.n+1:end] .= .- kkt.jacobian_scaling .* x
end

function set_jacobian_scaling!(kkt::DenseCondensedKKTSystem, constraint_scaling::AbstractVector)
    copyto!(kkt.jacobian_scaling, constraint_scaling)
end

function eval_jac_wrapper!(ipp::InteriorPointSolver, kkt::DenseCondensedKKTSystem, x::Vector{Float64})
    nlp = ipp.nlp
    cnt = ipp.cnt
    ns = length(ipp.ind_ineq)
    @trace(ipp.logger, "Evaluating constraint Jacobian.")
    jac = get_jacobian(kkt)
    cnt.eval_function_time += @elapsed jac_dense!(nlp,view(x,1:get_nvar(nlp)),jac)
    compress_jacobian!(kkt)
    cnt.con_jac_cnt+=1
    cnt.con_jac_cnt==1 && (is_valid(jac) || throw(InvalidNumberException()))
    @trace(ipp.logger,"Constraint jacobian evaluation started.")
    return jac
end

function eval_lag_hess_wrapper!(ipp::InteriorPointSolver, kkt::DenseCondensedKKTSystem, x::Vector{Float64},l::Vector{Float64};is_resto=false)
    nlp = ipp.nlp
    cnt = ipp.cnt
    @trace(ipp.logger,"Evaluating Lagrangian Hessian.")
    ipp._w1l .= l.*ipp.con_scale
    hess = get_hessian(kkt)
    cnt.eval_function_time += @elapsed hess_dense!(
        nlp, view(x,1:get_nvar(nlp)), ipp._w1l, hess;
        obj_weight = (get_minimize(nlp) ? 1. : -1.) * (is_resto ? 0.0 : ipp.obj_scale[]))
    compress_hessian!(kkt)
    cnt.lag_hess_cnt+=1
    cnt.lag_hess_cnt==1 && (is_valid(hess) || throw(InvalidNumberException()))
    return hess
end

