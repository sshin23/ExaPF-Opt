
Base.@kwdef struct AugLagOptions
    scaling::Bool = true
    max_iter::Int = 100
    max_inner_iter::Int = 1000
    rate::Float64 = 10.0
    ωtol::Float64 = 1.0
    ωtol_min::Float64 = 1.0e-5
    α0::Float64 = 1.0
    verbose::Int = 0
    inner_algo::Symbol = :tron
    lsq_lambda::Bool = false
    ε_primal::Float64 = 1e-8
    ε_dual::Float64 = 1e-8
end

struct AuglagSolver{InnerOptimizer} <: AbstractExaOptimizer
    optimizer::InnerOptimizer
    options::AugLagOptions
end

function solve_subproblem!(algo::AuglagSolver{<:MOI.AbstractOptimizer}, aug::AugLagEvaluator, uₖ)
    n_iter = aug.counter.gradient
    # Initiate optimizer
    MOI.empty!(algo.optimizer)
    MOI.set(algo.optimizer, MOI.RawParameter("tol"), algo.options.ωtol)
    # Pass the problem to the MOIEvaluator
    moi_solution = optimize!(algo.optimizer, aug, uₖ)
    return (
        status=moi_solution.status,
        iter=aug.counter.gradient - n_iter,
        minimizer=moi_solution.minimizer,
    )
end

# Augmented Lagrangian method
function optimize!(
    algo::AuglagSolver,
    aug::AugLagEvaluator,
    u0::AbstractVector;
)
    opt = algo.options
    nlp = aug.inner
    m = n_constraints(nlp)
    u♭, u♯ = bounds(nlp, Variables())

    # Initialize arrays
    uₖ        = copy(u0)
    u_start   = copy(u0)
    wk        = copy(u0)
    u_prev    = copy(u0)
    grad      = similar(u0) ; fill!(grad, 0)
    ut        = similar(u0) ; fill!(ut, 0)
    cons      = similar(u0, m) ; fill!(cons, 0)

    obj = Inf
    norm_grad = Inf

    tracer = Tracer()

    ρ0 = aug.ρ
    ωtol = opt.ωtol
    α0 = opt.α0
    verbose = (opt.verbose > 0)

    # Initialization (aka iteration 0)
    update!(aug, uₖ)
    # Get gradient of Augmented Lagrangian
    gradient!(aug, grad, uₖ)
    feasible_direction!(wk, wk, uₖ, grad, 1.0, u♭, u♯)

    ε_primal = opt.ε_primal
    ε_dual = opt.ε_dual * (1.0 + norm(wk))

    ηk = 1.0 / (ρ0^0.1)

    # Init multiplier
    if opt.lsq_lambda
        copy!(aug.λ, estimate_multipliers(aug, uₖ))
    end

    if verbose
        name = ""#MOI.get(algo.optimizer, MOI.SolverName())
        println("AugLag algorithm, running with $(name)\n")

        println("Total number of variables............................:      ", n_variables(nlp))
        println("Total number of constraints..........................:      ", n_constraints(nlp))
        println()

        log_header()
        # O-th iteration
        obj = objective(nlp, uₖ)
        primal_feas = primal_infeasibility!(nlp, cons, uₖ)
        dual_feas = norm(wk, 2)
        log_iter(0, obj, primal_feas, dual_feas, ηk, aug.ρ, 0)
    end

    local solution
    status = MOI.ITERATION_LIMIT
    mul = copy(aug.λ)

    tic = time()
    for i_out in 1:opt.max_iter
        uₖ .= u_start

        # Solve inner problem
        solution = solve_subproblem!(algo, aug, uₖ)

        uₖ = solution.minimizer
        n_iter = solution.iter

        # Update information w.r.t. original evaluator
        obj = objective(nlp, uₖ)
        # Get gradient of Augmented Lagrangian
        gradient!(aug, grad, uₖ)
        feasible_direction!(wk, wk, uₖ, grad, 1.0, u♭, u♯)

        # Primal feasibility
        primal_feas = norm(aug.cons, Inf)
        # Dual feasibility
        dual_feas = norm(wk, 2)

        if (dual_feas < ε_dual) && (primal_feas < ε_primal)
            status = MOI.OPTIMAL
            break
        end

        # Update starting point
        u_start .= uₖ
        # Update the penalties (see Nocedal & Wright, page 521)
        if primal_feas <= ηk
            update_multipliers!(aug)
            mul = hcat(mul, aug.λ)
            ηk = ηk / (aug.ρ^0.9)
            # ωtol /= aug.ρ
            # ωtol = max(ωtol, opt.ωtol_min)
        else
            update_penalty!(aug; η=opt.rate)
            ηk = 1.0 / (aug.ρ^0.1)
            # ωtol = 1.0 / aug.ρ
            # ωtol = max(ωtol, opt.ωtol_min)
        end

        # Log
        verbose && log_iter(i_out, obj, primal_feas, dual_feas, ηk, aug.ρ, n_iter) # Log evolution
        push!(tracer, obj, primal_feas, dual_feas)
    end
    toc = time() - tic

    if verbose
        println()
        println("Number of iterations....: ", length(tracer.objective))
        println("Number of objective function evaluations             = ", aug.counter.objective)
        println("Number of objective gradient evaluations             = ", aug.counter.gradient)
        println("Number of Lagrangian Hessian evaluations             = ", aug.counter.hessian)
        @printf("Total CPU time                                       = %.3f\n", toc)
        println()
    end

    solution = (
        status=status,
        minimum=obj,
        minimizer=uₖ,
        trace=tracer,
        multipliers=mul,
    )

    return solution
end

