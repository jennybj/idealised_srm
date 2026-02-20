# Load distributed computing libraries with SLURM
using Distributed, SlurmClusterManager
addprocs(SlurmManager())

@everywhere begin
    # ===========
    # Load packages
    # ===========
    using Optim, Roots,LinearAlgebra, Statistics,Interpolations, Plots, DelimitedFiles, CSV, DataFrames, Formatting, BenchmarkTools
    using FastGaussQuadrature, ProgressMeter
    include("io3.jl")
    include("creategrid1.jl")


    # ============
    # Base Set up
    # ============
    start_year = 1990
    end_year = 2140
    T_horizon = end_year - start_year
    geo_start_year = 2030
    geo_start = geo_start_year-start_year + 1
    geo_length = T_horizon - geo_start + 1

    input_path =  "/gpfs/gibbs/project/tsmith/jhc84/Geoengineering/Input Files/"
    output_path = "/gpfs/gibbs/project/tsmith/jhc84/Geoengineering/Output Files/"
    decision_path = "/gpfs/gibbs/project/tsmith/jhc84/Geoengineering/Input Files/Decision Rules/"

end

@everywhere begin
    # ===========
    # Parameters
    # ===========

    emis_scaled = readdlm(open(output_path * "emissions.txt", "r"), skipstart=0)[:, 2]

    const chi_horizon = 150 # For t>t_horizon, χ(t) = 0
    const gn_horizon = 150 # For t> gn_horizon, gn = 0.0

    if chi_horizon > gn_horizon
        simul_time = chi_horizon
    else
        simul_time = gn_horizon
    end
    const N_w = 21    # wealth grid size
    const N_z = 21    # shock grid size
    const ncells = 19240  # number of regions
    const ga = 0.015  # TFP growth
    const β = 0.985   # discount factor
    const δ = 0.06    # depreciation
    const α = 0.36    # capital share
    const energyshare = 0.062
    const rss = (1 + ga) / β - 1
    const θ = 1 / (1 + energyshare)
    const b = 0.4     # production scaling

    # Emission levels for year 1 calibration
    const cumstock1990 = 216.8650
    const cumstock1991 = 222.9089
    escale = 1e3 # energy scaler
    yscale = 1e-3 # output scaler

    # ===========
    # Load Input Files
    # ===========
    coef_file = "coefficients_and_RMSE_v1.txt"
    io = open(input_path * coef_file, "r")
    datamatrix = readdlm(io)
    close(io)


    # γ_1 and γ_2 are temperature sensitivity parameters
    γ_1 = datamatrix[:, 4]
    γ_2 = datamatrix[:, 5]

    # ρ and ϵ are region-specific AR(1) shock parameters
    ρ = datamatrix[:, 6] 
    ϵ = datamatrix[:, 7]

    # Load pre-industrial temperatures by region
    T_preind = readdlm(open(input_path * "picontrol_v1.txt"), skipstart=0)[:, 4]

    # =================
    # Greening Function
    # =================
    η_0001, η_05 = 10, 75
    χ(t) = inv(1 + exp(log(0.01 / 0.99) * (t - η_05) / (η_0001 - η_05)))
    pctdirty = χ.(creategrid(1, 200, 200)) / χ(1)

    # Load Solar Radiation Management coefficients
    SRM_file = "SRM_coefficients_v2.txt"
    io = open(input_path* SRM_file, "r")
    datamatrix = readdlm(io, skipstart = 8)
    ζ_1 = datamatrix[:,3]
    ζ_2 = datamatrix[:,4]
    offset = datamatrix[:,5]

    # Load region-specific parameters
    countryi = fill("",ncells)
    lati = fill(0,ncells)
    loni = fill(0,ncells)
    areai = fill(0.0,ncells)
    rigi = fill(0.0,ncells)
    avgtempi = fill(0.0,ncells)
    popregi1990 = fill(0.0,ncells)
    gdpregi1990 = fill(0.0,ncells)
    gdpperregi1990 = fill(0.0,ncells)
    dump = fill(0.0,ncells)

    filename = input_path*"parse2.gin6"
    io = open(filename,"r")
    for i in 1:ncells
    dump[i],lati[i],loni[i],countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
        gdpregi1990[i],gdpperregi1990[i] = readio(io,(3,"b3","a40",6))
    end

    # Load Population Coefficients
    io = open(input_path*"regpop4.pop", "r")
    datamatrix = readdlm(io, skipstart = 0)
    close(io)

    popi = datamatrix[:,4:end]

    io = open(input_path*"regpop4.grate", "r")
    datamatrix = readdlm(io, skipstart = 0)
    close(io)
    gn_mat = datamatrix[:,4:end]

    # Additional Parameters
    cum_emissions = readdlm(open(input_path * "cumulative_emissions_global_temperature_v1.txt", "r"), skipstart=4)[:, 2]

    # Compute annual emissions from cumulative path
    orig_emissions = [cum_emissions[i + 1] - cum_emissions[i] for i in 1:(length(cum_emissions) - 1)]
    append!(orig_emissions, fill(orig_emissions[end], 40))  # extend with flat tail


    # Rescale GDP and compute emissions per dirty energy unit
    gdpnetperi = round.(gdpperregi1990 * yscale, digits = 8)
    globalgdp1990 = sum(gdpregi1990)
    x1990 = escale * orig_emissions[141] / pctdirty[1] 
    const p = energyshare * globalgdp1990 / x1990  # price of energy (fixed)

    capitali = α * gdpnetperi / (rss + δ)
    xi = ((1 - θ) / (θ * p)) * gdpnetperi
    ai = (b * (1 - θ) * (capitali .^ (α * θ)) .* (xi .^ (-θ)) / p) .^ (1 / (θ * (α - 1)))

    # Steady-state capital (for representative agent = 1)
    function calcki(ai)
        return ((((rss + δ) / (α * θ)) * (b^(-1 / θ)) * ((p / (1 - θ))^((1 - θ) / θ))) ^ (1 / (α - 1))) * ai
    end
    k_ss = calcki(1)

    # ==================================
    # Functions (see Notes for additional details)
    # ==================================
    # Helper Function to generate the three grids used in code
    function generate_grids(ρ, ϵ, T; N_w=N_w, N_z=N_z, k_ss=k_ss)
        k_grid = creategrid(0.5 * k_ss, 1.5 * k_ss, N_w)
        stdev = (ϵ == 0) ? 1.0 : sqrt(ϵ^2 / (1 - ρ^2))
        z_grid = creategrid(-3 * stdev, 3 * stdev, N_z)
        w_grid = creategrid(0.5 * G(k_ss, 0, T), 1.5 * G(k_ss, 0, T), N_w)
        return k_grid, z_grid, w_grid
    end

    # Damage Function
    function D(t; α = α) 
        T = 12.609
        d = 0.02
        κ_plus = 0.00362887
        κ_minus =  0.00327721
        if t <= T
            return ((1-d) * exp(-κ_minus*(t-T)^2) + d)^(1/(1-α))
        else return  ((1-d) * exp(-κ_plus*(t-T)^2) + d)^(1/(1-α))
        end
    end

    d(T1, T2) = D(T2)/D(T1) # damage from year-to-year transition in temperature
    d_shock(T, z) = D(T+z) / D(T) # damage from stochastic temperature shock
    h(k, z, T) = ((1-θ)*b/p)^(1/θ) * k^α * d_shock(T, z)^(1-α) # energy choice function

    # Steady-state energy chocie x_ss. Note that x_ss = x_0 and k_ss = k_0 for the forward simulation
    x_ss = h(k_ss, 0, 10.2)

    # Production function
    F(k, x, z, T) = b * k^(α*θ) * d_shock(T, z)^(θ - α*θ)*x^(1-θ) - p*x

    
    # Wealth function
    G(k, z, T) = F(k, h(k, z, T), z, T) + (1-δ)*k
    
    # Partial Derivative of wealth function
    partial_G_k(k, z, T) = α  * b * k^(α-1) * d_shock(T, z)^(1-α)*((1-θ)*b/p)^((1-θ)/θ) - p*α*((1-θ)*b/p)^(1/θ)*k^(α-1)*d_shock(T, z)^(1-α)+ (1-δ)
    
    function get_temp_pre_srm(carbonstock, γ_1, γ_2, T_preind)
        temp = T_preind + γ_1 * carbonstock + γ_2 * carbonstock^2
        return temp
    end

    function get_temp_post_srm(γ_1, γ_2, T_preind, ζ_1, ζ_2,  carbonstock, srmyear)
        temp = get_temp_pre_srm(carbonstock, γ_1, γ_2, T_preind) + ζ_1*(1 - exp(ζ_2*srmyear))
        return temp
    end

    # ===============================
    # Irregulat Wealth Grid calculation
    # ===============================
    function irreg_w_grid(H, k_grid, T1, T2, nregion, nyear; N_w=N_w, N_z=N_z)
        w = similar(H)
        if nyear >= gn_horizon
            gn = 0.0
        else
            gn = gn_mat[nregion, nyear]
        end
        dT = d(T1, T2)
        for j in 1:N_w, l in 1:N_z
            w[j, l] = β^(-1)*(1 + ga)*dT*inv(H[j, l])+(1 + gn)*(1 + ga)*dT*k_grid[j]
        end
        return w
    end

    # ===============================
    # Regularization of wealth grid 
    # ===============================
    function reg_w_grid(w_irreg_grid, w_grid, k_grid)
        L = size(w_irreg_grid, 2)
        k_prime = similar(w_irreg_grid)
        for l in 1:L
            itp = linear_interpolation(w_irreg_grid[:, l], k_grid, extrapolation_bc=Line())
            k_prime[:, l] .= itp.(w_grid)
        end
        return k_prime
    end

    # ==================
    # Read Decision rules for pre-SRM path
    # ==================
    function readrules(i)
        year = string(start_year+i-1)
        rulematrix = CSV.read(decision_path*"decrule_"*year*".csv", DataFrame)
        rulek = zeros(ncells, N_w, N_z)
        for i in 1:ncells
            endrow = N_w*i 
            beginrow = N_w*(i-1) +1
            for j in 1:N_z
                zcol = rulematrix[!, j+3]
                rulek[i, :, j] = zcol[ beginrow:endrow]
            end
            
        end
        return rulek
    end

    # ===============================
    # Interpolated policy function ĥ(w, z)
    # ===============================
    function h_hat(w, z, kprime, w_grid, z_grid)
        interp = interpolate((w_grid, z_grid), kprime, Gridded(Linear()))
        extp = extrapolate(interp, Line())
        val = extp(w, z)
        if val <= 0
            val = 0.001
        end

        return val  # Handles both in and out-of-bounds cases
    end

    # ===============================
    # Value function (Lambda)
    # ===============================
    function Λ(k, z, kprime, w_grid, z_grid, T1, T2, nregion, nyear)
        
        if nyear >= gn_horizon
            gn  = 0.0
        else
            gn = gn_mat[nregion, nyear]
        end
        Gval = G(k, z, T1)
        hval = h_hat(Gval, z, kprime, w_grid, z_grid)
        return inv(Gval - (1 + ga) * (1 + gn) * d(T1, T2) * hval)
    end

    # ===============================
    # Expectation update for H via quadrature
    # ===============================
    function H_update(ρ, ϵ, kprime, k_grid, w_grid, z_grid, T1, T2, nregion, nyear, abscissa, weights; N_w=N_w, N_z=N_z)
        H1 = zeros(N_w, N_z)
        M = length(abscissa)

        if ϵ == 0
            for j in 1:N_w, l in 1:N_z
                z_val = z_grid[l] * ρ
                H1[j, l] = Λ(k_grid[j], z_val, kprime, w_grid, z_grid, T1, T2, nregion, nyear) *
                           partial_G_k(k_grid[j], z_val, T1)
            end
        else
            z_i_m = zeros(M)
            λ_vec = zeros(M)
            for j in 1:N_w, l in 1:N_z
                for m in 1:M
                    z_i_m[m] = ρ * z_grid[l] + sqrt(2) * ϵ * abscissa[m]
                    λ_vec[m] = Λ(k_grid[j], z_i_m[m], kprime, w_grid, z_grid, T1, T2, nregion, nyear)
                end
                G_partials = partial_G_k.(k_grid[j], z_i_m, T1)
                H1[j, l] = π^(-0.5) * sum(weights .* λ_vec .* G_partials)
            end
        end
        return H1
    end

    # ===============================
    # Steady-state routine for backwards iteration starting point
    # ===============================
    function compute_steady_state(M, ρ, ϵ, T, nregion; maxiter=500, toler=1e-8, nyear = 151)
        err = Inf
        niter = 0

        k_grid, z_grid, w_grid = generate_grids(ρ, ϵ, T)
        abscissa, weights = gausshermite(M)
        H0 = ones(N_w, N_z)
        H1 = similar(H0)
        kprime = nothing

        while err > toler && niter < maxiter
            niter += 1

            # Step 1: Compute irregular wealth grid
            irreg_w = irreg_w_grid(H0, k_grid, T, T, nregion, nyear)

            # Step 2: Interpolate to regular grid
            kprime = reg_w_grid(irreg_w, w_grid, k_grid)

            # Step 3: Update H
            H1 .= H_update(ρ, ϵ, kprime, k_grid, w_grid, z_grid, T, T, nregion, nyear, abscissa, weights)

            # Step 4: Compute error and update guess
            err = maximum(abs.(H1 .- H0))
            H0 .= H1
        end
        return H1, kprime
    end


    # ===============================
    # Backward Iteration
    # ===============================
    function iterate_backwards(emissions, M, γ_1, γ_2, ρ, ϵ, T_preind, ζ_1, ζ_2, offset, nregion; nyears=T_horizon)
        abscissa, weights = gausshermite(M)
        k_grid, z_grid, w_grid = generate_grids(ρ, ϵ, T_preind)
        temp_path = zeros(nyears+1)
        # Compute carbon stock path from emissions
        carbon_stock = cumsum(vcat(cumstock1990, emissions))

        # Calculate regional temperature from carbon stock
        for i in 1:nyears+1
            if i < geo_start
                temp_path[i] = get_temp_pre_srm(carbon_stock[i], γ_1, γ_2, T_preind)
            elseif i <= 140
                temp_path[i] = get_temp_post_srm(γ_1, γ_2, T_preind, ζ_1, ζ_2, carbon_stock[i], i-geo_start)
            else
                temp_path[i] = get_temp_pre_srm(carbon_stock[i], γ_1, γ_2, T_preind) + offset
            end
        end
        
        # Allocate output arrays
        H_matrix = zeros(N_w, N_z, nyears+1)
        kprime_matrix = zeros(N_w, N_z, nyears+1)

        # Final period steady state
        H_matrix[:, :, end], kprime_matrix[:, :, end] = compute_steady_state(M, ρ, ϵ, temp_path[end], nregion)

        # Temporary arrays for quadrature
        z_i_m = zeros(M)
        λ_vec = similar(z_i_m)
        G_vals = similar(z_i_m)

        # Backward pass
        for t in nyears:-1:1
            T1, T2 = temp_path[t], temp_path[t+1]

            irreg_w = irreg_w_grid(H_matrix[:, :, t+1], k_grid, T1, T2, nregion, t)
            kprime_matrix[:, :, t] = reg_w_grid(irreg_w, w_grid, k_grid)

            # Quadrature Routin
            for j in 1:N_w, l in 1:N_z
                for m in 1:M
                    z_i_m[m] = ρ * z_grid[l] + sqrt(2) * ϵ * abscissa[m]
                    λ_vec[m] = Λ(k_grid[j], z_i_m[m], kprime_matrix[:, :, t], w_grid, z_grid, T1, T2, nregion, t)
                    G_vals[m] = partial_G_k(k_grid[j], z_i_m[m], T1)
                end
                # Update on H using quadrature
                H_matrix[j, l, t] = π^(-0.5) * sum(weights .* λ_vec .* G_vals)
            end
        end

        "Backwards Iteration Done for Region $nregion"
        return H_matrix, kprime_matrix
    end

    # ===============================
    # Forward Simulation
    # ===============================
    function simulate_forward(kprime_mat, expected_emissions; nyears=T_horizon)
        # Proportions is used to calculate area-weighted average temperature
        proportions = areai .* rigi ./ sum(areai .* rigi)
        pop_proportions = zeros(ncells, nyears+1)

        for i in 1:nyears+1
            pop_proportions[:,i] =  popi[:,i]./ sum(popi[:,i])
        end

        # Pre-allocate large arrays
        carbon_stock = zeros(nyears+1)
        expected_carbonstock = vcat(cumstock1990, cumstock1990 .+ cumsum(expected_emissions))
        average_temp = zeros(nyears+1)
        pop_temp = zeros(nyears+1)
        carbon_stock[1] = cumstock1990

        expected_temp = zeros(ncells, nyears+1)
        emissions = zeros(ncells, nyears)
        total_emissions = zeros(nyears)

        k = fill(k_ss, ncells, nyears+1)
        x = fill(x_ss, ncells, nyears+1)
        e = zeros(ncells, nyears+1)
        w = zeros(ncells, nyears)
        y = zeros(ncells, nyears)

        reg_gdp = zeros(ncells, nyears)
        reg_gdp_unsc = zeros(ncells, nyears)
        gdp = zeros(nyears)
        gdp_unsc = zeros(nyears)
        reg_temp = zeros(ncells, nyears+1)
        actual_damages = zeros(ncells, nyears+1)


        # Calculate Expected Temperature and related damages

        for i in 1:nyears+1
            for j in 1:ncells
                if i < geo_start
                    expected_temp[j,i] = get_temp_pre_srm(expected_carbonstock[i], γ_1[j], γ_2[j], T_preind[j])
                elseif i <= 140
                    expected_temp[j,i] = get_temp_post_srm(γ_1[j], γ_2[j], T_preind[j], ζ_1[j], ζ_2[j], expected_carbonstock[i], i - geo_start)
                else 
                    expected_temp[j,i] = get_temp_pre_srm(expected_carbonstock[i], γ_1[j], γ_2[j], T_preind[j]) .+ offset[j]
                end
            end
        end
        expected_damages = D.(expected_temp)
        
        # Calculate Productivity Path
        a_i = similar(expected_temp)
        a_i[:, 1] .= ai
        for i in 2:nyears+1
            a_i[:, i] .= (1 + ga) .* expected_damages[:, i] ./ expected_damages[:, i - 1] .* a_i[:, i - 1]
        end
        
        old_rules = zeros(ncells, N_w, N_z, geo_start -1)
        for i in 1:geo_start-1
            old_rules[:,:,:,i] = readrules(i)
        end
        println("Read in Old Rules")

        @showprogress "Simulating Forward" for i in 1:nyears

            emissions[:, i] .= a_i[:, i] .* popi[:,i] .* x[:, i] .* pctdirty[i] ./ escale # Unscale Emissions
            total_emissions[i] = sum(emissions[:, i])
            carbon_stock[i+1] = carbon_stock[i] + total_emissions[i] # Update Carbonstock
            if i < geo_start
                reg_temp[:, i] .= get_temp_pre_srm.(carbon_stock[i], γ_1, γ_2, T_preind)
            elseif i <= 140
                reg_temp[:, i] .= get_temp_post_srm.(γ_1, γ_2, T_preind, ζ_1, ζ_2, carbon_stock[i], i - geo_start) #Calculate Regional Temeprature
            else 
                reg_temp[:, i] .= get_temp_pre_srm.(carbon_stock[i], γ_1, γ_2, T_preind) .+ offset
            end
            average_temp[i] = sum(proportions .* reg_temp[:, i]) # Calculate area-weighted average temperature
            actual_damages[:, i] .= D.(reg_temp[:, i]) # Calculate damages from experienced (not expected) temperature

            for j in 1:ncells
                T1 = reg_temp[j, i] # Experienced temperature
                #z = reg_temp[j, i] - expected_temp[j, i] # Stochastic Shock
                z = 0
                y[j, i] = F(k[j, i], h(k[j, i], z, T1), z, T1) # Update output
                w[j, i] = G(k[j, i], z, T1) # Update Wealth
                reg_gdp[j, i] = y[j, i] * a_i[j, i] * popi[j,i] # Unscaled Regional GDP
                reg_gdp_unsc[j, i] = y[j,i] * a_i[j,1]# Scaled Regional GDP
            end

            gdp[i] = sum(reg_gdp[:, i]) # Unscaled Global GDP
            gdp_unsc[i] = sum(reg_gdp_unsc[:, i]) # Scaled Global GDP

            for j in 1:ncells
                z = 0
                T1 = reg_temp[j, i]
                k_grid, z_grid, w_grid = generate_grids(ρ[j], ϵ[j], T1)

                if i < geo_start
                    kprime = old_rules[j, :, :, i]
                else
                    kprime = kprime_mat[j, :, :, i] # Select correct array for policy function
                end

                k[j,i+1] = h_hat(w[j,i], z, kprime, w_grid, z_grid) # Interpolate savings decision
                x[j,i+1] = h(k[j,i], z, T1) # Expected energy use next year
            end
        end
        return carbon_stock, average_temp, total_emissions, emissions, reg_temp, reg_gdp, reg_gdp_unsc, a_i, k, w
    end

    # ===============================
    # Iterative emissions adjustment
    # ===============================
    function iterate(M, γ_1, γ_2, ρ, ϵ, T_preind, ζ_1, ζ_2, emis_scaled; maxiter=500, toler=1e-8, nyears = T_horizon)
        # Pre-allocate large arrays
        H = zeros(ncells, N_w, N_z, nyears+1)
        kprime = zeros(ncells, N_w, N_z, nyears+1)
        emiss_0 = copy(emis_scaled)
        carbonstock = zeros(nyears+1)
        average_temp = zeros(nyears)
        pop_temp = zeros(nyears)
        total_emissions = zeros(nyears)
        reg_temp = zeros(ncells, nyears)
        emiss_0 = copy(emis_scaled)

        reg_gdp = zeros(ncells, nyears)
        reg_gdp_fp = zeros(ncells, nyears)
        reg_emissions = zeros(ncells, nyears)

        a_i = zeros(ncells, nyears)
        k = zeros(ncells, nyears+1)
        w = zeros(ncells, nyears)

        niter = 0
        err = Inf

        while niter <= maxiter && err > toler
            niter += 1
            # Parallel function call
            res = @showprogress pmap(i -> iterate_backwards(emiss_0, M, γ_1[i], γ_2[i], ρ[i], ϵ[i], T_preind[i], ζ_1[i], ζ_2[i], offset[i], i), 1:ncells)
            println("Iterated Backwards for all regions!")
            # Read parallel function return into usable format
            for i in 1:ncells
                H[i, :, :, :] .= res[i][1]
                kprime[i, :, :, :] .= res[i][2]
            end

            # Simulate forward to calculate new emissions trajectory
            carbonstock, average_temp, total_emissions, reg_emissions, reg_temp, reg_gdp, reg_gdp_fp, a_i, k, w = simulate_forward(kprime, emiss_0)
            
            # Calculate error
            err = maximum(abs.(emiss_0 .- total_emissions))
            println("Error: $err")

            # Update guess
            emiss_0 .= total_emissions
        end

        return carbonstock, average_temp, total_emissions, reg_emissions, reg_temp, H, kprime, reg_gdp, reg_gdp_fp, a_i, k, w
    end

end

# Fixed Point Iteration

#@time compute_steady_state(5, ρ[1], ϵ[1], T_preind[1], 1)
#pmap(i -> iterate_backwards(emis_scaled, 5, γ_1[i], γ_2[i], ρ[i], ϵ[i], T_preind[i], ζ_1[i], ζ_2[i], i), 1:ncells)


carbonstock, average_temp, total_emissions, reg_emissions, reg_temp, H, kprime, reg_gdp, reg_gdp_fp, a_i, k, w = iterate(5, γ_1, γ_2, ρ, ϵ, T_preind, ζ_1, ζ_2, emis_scaled)

# Write output files


io = open(output_path*"ai.txt", "w")
writearrays(io, (12 ,18.8), a_i)
close(io)

io = open(output_path*"chi.txt", "w")
writearrays(io, (6 ,18.8), pctdirty)
close(io)

io = open(output_path*"capital.txt", "w")
writearrays(io, (12 ,18.8), k)
close(io)

io = open(output_path*"wealth.txt", "w")
writearrays(io, (12 ,18.8), w)
close(io)

io = open(output_path*"reg_gdp.txt", "w")
writearrays(io, (12 ,18.8), reg_gdp)
close(io)

io = open(output_path*"reg_gdp_fp.txt", "w")
writearrays(io, (12 ,18.8), reg_gdp_fp)
close(io)

io = open(output_path*"regional_emissions.txt", "w")
writearrays(io, (12, 15.8), reg_emissions)
close(io)

io = open(output_path*"emissions.txt", "w")
writearrays(io, (8 ,15.8), total_emissions)
close(io)


io = open(output_path*"average_temperature.txt", "w")
writearrays(io, (8, 15.8), average_temp)
close(io)

io = open(output_path*"regional_temperature.txt", "w")
writearrays(io, (12, 15.8), reg_temp)
close(io)

io = open(output_path*"carbonstock.txt", "w")
writearrays(io, (18, 15.8), carbonstock)
close(io)



function write_to_csv(data::Array{Float64, 4}; ncells = ncells, nyears = 150)    
    w_grid = creategrid(0.5 * G(k_ss, 0, 0), 1.5*G(k_ss, 0, 0), N_w)
    rules_output = output_path*"decision_rules/"
    column_names = ["latitude", "longitude", "w"]  # Add new columns
    append!(column_names, ["z$i" for i in 1:N_z])
    # Iterate over the 4th dimension (150 slices)
    for t in 1:nyears
        # Extract the 19240 × 21 × 21 slice for the current `t`
        slice = round.(data[:, :, :, t], digits=8)
        
        # Create a DataFrame to represent the entire structure
        rows = []
        for i in 1:ncells # Iterate over the 20249 matrices
            matrix = slice[i, :, :] # Extract the 21 × 21 matrix
            for row in 1:N_w
                w_value = w_grid[row]
                push!(rows, (lati[i], loni[i], w_value, vec(matrix[row, :])...)) # Add each row of the matrix as a separate row
            end
        end
        # Convert the collected rows to a DataFrame
        df = DataFrame(rows, column_names)
        
        # Generate file name
        file_name = joinpath(rules_output, "decrule_$(t+1989).csv")
        
        # Write the DataFrame to a CSV file
        CSV.write(file_name, df)
    end
end

#write_to_csv(kprime)