include("plot_helper.jl")


# Set each path to the corresponding folder on your machine.
#   output_path       — where figures will be saved
#   input_path        — folder containing the input files:
#                         parse2.gin6, regpop4.pop, picontrol_v1.txt,
#                         SRM_coefficients_v2.txt, coefficients_and_RMSE_v1.txt
#   couple_path_nosrm — folder containing coupled model output without SRM
#                         (subfolders e1/, e2/, e3/ with output_year_*.txt files)
#   couple_path_srm   — folder containing coupled model output with SRM
#                         (subfolders e1/, e2/, e3/ with output_year_*.txt files)


output_path       = ""  
input_path        = ""  
couple_path_nosrm = ""   
couple_path_srm   = ""  


const ncells = 19240  # number of regions
const ga = 0.015  # TFP growth
const β = 0.985   # discount factor
const δ = 0.06    # depreciation
const α = 0.36    # capital share
const energyshare = 0.062
const rss = (1 + ga) / β - 1
const θ = 1 / (1 + energyshare)
const b = 0.4
escale = 1e3 # energy scaler
yscale = 1e-3 # output scaler

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

filename = input_path * "parse2.gin6"
io = open(filename,"r")
for i in 1:ncells
dump[i],lati[i],loni[i],countryi[i],areai[i],rigi[i],avgtempi[i],popregi1990[i],
    gdpregi1990[i],gdpperregi1990[i] = readio(io,(3,"b3","a40",6))
end

gdpnetperi = round.(gdpperregi1990 * yscale, digits = 8)
globalgdp1990 = sum(gdpregi1990)
η_0001, η_05 = 10, 75
χ(t) = inv(1 + exp(log(0.01 / 0.99) * (t - η_05) / (η_0001 - η_05)))
pctdirty = χ.(creategrid(1, 200, 200)) / χ(1)
x1990 = escale * (216.8650-210.7950)/pctdirty[1]
const p = energyshare * globalgdp1990 / x1990
# Load Population Coefficients
io = open(input_path * "regpop4.pop", "r")
datamatrix = readdlm(io, skipstart = 0)
close(io)

popi = datamatrix[:,4:end]
glob_pop = vec(sum(popi, dims = 1))
glob_pop_10 = glob_pop .* 0.1
glob_pop_90 = glob_pop .* 0.9

SRM_file = "SRM_coefficients_v2.txt"
io = open(input_path* SRM_file, "r")
datamatrix = readdlm(io, skipstart = 8)
ζ_1 = datamatrix[:,3]
ζ_2 = datamatrix[:,4]
offset = Float64.(datamatrix[:,5])

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

h(k, z, T) = ((1-θ)*b/p)^(1/θ) * k^α * d_shock(T, z)^(1-α)
F(k, x, z, T) = b * k^(α*θ) * d_shock(T, z)^(θ - α*θ)*x^(1-θ) - p*x

function calc_output(k, z, T, ai)
    x = h(k, z, T)
    y = F(k, x, z, T) * ai
    return y

end

function load_simulation_data(couple_path, e1_prefix, e2_prefix, e3_prefix, n_simul, ncells, ga)
    k_e1, k_e2, k_e3 = zeros(ncells, n_simul), zeros(ncells, n_simul), zeros(ncells, n_simul)
    e_e1, e_e2, e_e3 = zeros(ncells, n_simul), zeros(ncells, n_simul), zeros(ncells, n_simul)
    ai_e1, ai_e2, ai_e3 = zeros(ncells, n_simul), zeros(ncells, n_simul), zeros(ncells, n_simul)
    z_e1, z_e2, z_e3 = zeros(ncells, n_simul), zeros(ncells, n_simul), zeros(ncells, n_simul)
    temp_exp_e1, temp_exp_e2, temp_exp_e3 = zeros(ncells, n_simul), zeros(ncells, n_simul), zeros(ncells, n_simul)
    gdp_e1, gdp_e2, gdp_e3 = zeros(ncells, n_simul), zeros(ncells, n_simul), zeros(ncells, n_simul)

    for i in 1:n_simul
        year = 2029 + i
        e1 = readdlm(open(couple_path * e1_prefix * "output_year_$year.txt", "r"), skipstart = 15)
        e2 = readdlm(open(couple_path * e2_prefix * "output_year_$year.txt", "r"), skipstart = 15)
        e3 = readdlm(open(couple_path * e3_prefix * "output_year_$year.txt", "r"), skipstart = 15)

        k_e1[:, i] = e1[:, 7]
        k_e2[:, i] = e2[:, 7]
        k_e3[:, i] = e3[:, 7]

        e_e1[:, i] = e1[:, 14]
        e_e2[:, i] = e2[:, 14]
        e_e3[:, i] = e3[:, 14]

        ai_e1[:, i] = e1[:, 9]
        ai_e2[:, i] = e2[:, 9]
        ai_e3[:, i] = e3[:, 9]

        z_e1[:, i] = e1[:, 5]
        z_e2[:, i] = e2[:, 5]
        z_e3[:, i] = e3[:, 5]

        temp_exp_e1[:, i] = e1[:, 3]
        temp_exp_e2[:, i] = e2[:, 3]
        temp_exp_e3[:, i] = e3[:, 3]

        gdp_e1[:, i] = calc_output.(k_e1[:, i], z_e1[:, i], temp_exp_e1[:, i], ai_e1[:, i]) / (1 + ga)^(i - 1)
        gdp_e2[:, i] = calc_output.(k_e2[:, i], z_e2[:, i], temp_exp_e2[:, i], ai_e2[:, i]) / (1 + ga)^(i - 1)
        gdp_e3[:, i] = calc_output.(k_e3[:, i], z_e3[:, i], temp_exp_e3[:, i], ai_e3[:, i]) / (1 + ga)^(i - 1)

    end

    return (k_e1=k_e1, k_e2=k_e2, k_e3=k_e3,
            e_e1=e_e1, e_e2=e_e2, e_e3=e_e3,
            ai_e1=ai_e1, ai_e2=ai_e2, ai_e3=ai_e3,
            z_e1=z_e1, z_e2=z_e2, z_e3=z_e3,
            temp_exp_e1=temp_exp_e1, temp_exp_e2=temp_exp_e2, temp_exp_e3=temp_exp_e3,
            gdp_e1=gdp_e1, gdp_e2=gdp_e2, gdp_e3=gdp_e3)
end

n_simul = 71

# Load nosrm data
data_nosrm = load_simulation_data(couple_path_nosrm, "e1/", "e2/", "e3/", n_simul, ncells, ga)
k_e1_nosrm, k_e2_nosrm, k_e3_nosrm = data_nosrm.k_e1, data_nosrm.k_e2, data_nosrm.k_e3
e_e1_nosrm, e_e2_nosrm, e_e3_nosrm = data_nosrm.e_e1, data_nosrm.e_e2, data_nosrm.e_e3
ai_e1_nosrm, ai_e2_nosrm, ai_e3_nosrm = data_nosrm.ai_e1, data_nosrm.ai_e2, data_nosrm.ai_e3
z_e1_nosrm, z_e2_nosrm, z_e3_nosrm = data_nosrm.z_e1, data_nosrm.z_e2, data_nosrm.z_e3
temp_exp_e1_nosrm, temp_exp_e2_nosrm, temp_exp_e3_nosrm = data_nosrm.temp_exp_e1, data_nosrm.temp_exp_e2, data_nosrm.temp_exp_e3
gdpcap_e1_nosrm, gdpcap_e2_nosrm, gdpcap_e3_nosrm = data_nosrm.gdp_e1, data_nosrm.gdp_e2, data_nosrm.gdp_e3
gdp_e1_nosrm, gdp_e2_nosrm, gdp_e3_nosrm = gdpcap_e1_nosrm.*popi[:,41:111], gdpcap_e2_nosrm.*popi[:,41:111], gdpcap_e3_nosrm.*popi[:,41:111]

# Load srm data
data_srm = load_simulation_data(couple_path_srm, "e1/2028_", "e2/2029_", "e3/2030_", n_simul, ncells, ga)
k_e1_srm, k_e2_srm, k_e3_srm = data_srm.k_e1, data_srm.k_e2, data_srm.k_e3
e_e1_srm, e_e2_srm, e_e3_srm = data_srm.e_e1, data_srm.e_e2, data_srm.e_e3
ai_e1_srm, ai_e2_srm, ai_e3_srm = data_srm.ai_e1, data_srm.ai_e2, data_srm.ai_e3
z_e1_srm, z_e2_srm, z_e3_srm = data_srm.z_e1, data_srm.z_e2, data_srm.z_e3
temp_exp_e1_srm, temp_exp_e2_srm, temp_exp_e3_srm = data_srm.temp_exp_e1, data_srm.temp_exp_e2, data_srm.temp_exp_e3
gdpcap_e1_srm, gdpcap_e2_srm, gdpcap_e3_srm = data_srm.gdp_e1, data_srm.gdp_e2, data_srm.gdp_e3
gdp_e1_srm, gdp_e2_srm, gdp_e3_srm = gdpcap_e1_srm.*popi[:,41:111], gdpcap_e2_srm.*popi[:,41:111], gdpcap_e3_srm.*popi[:,41:111]

function calc_decadal_temp_avg(temp_exp_e1, temp_exp_e2, temp_exp_e3, z_e1, z_e2, z_e3, year_range)
    ncells = size(temp_exp_e1, 1)
    avg_temp = zeros(ncells)
    for i in 1:ncells
        temp_e1 = mean(temp_exp_e1[i, year_range] .+ z_e1[i, year_range])
        temp_e2 = mean(temp_exp_e2[i, year_range] .+ z_e2[i, year_range])
        temp_e3 = mean(temp_exp_e3[i, year_range] .+ z_e3[i, year_range])
        avg_temp[i] = (temp_e1 + temp_e2 + temp_e3) / 3
    end
    return avg_temp
end

# Calculate decadal averages for nosrm
temp_30_10yr_avg_nosrm = calc_decadal_temp_avg(temp_exp_e1_nosrm, temp_exp_e2_nosrm, temp_exp_e3_nosrm,
                                               z_e1_nosrm, z_e2_nosrm, z_e3_nosrm, 1:10)
temp_90_10yr_avg_nosrm = calc_decadal_temp_avg(temp_exp_e1_nosrm, temp_exp_e2_nosrm, temp_exp_e3_nosrm,
                                               z_e1_nosrm, z_e2_nosrm, z_e3_nosrm, 61:71)

# Calculate decadal averages for srm
temp_30_10yr_avg_srm = calc_decadal_temp_avg(temp_exp_e1_srm, temp_exp_e2_srm, temp_exp_e3_srm,
                                             z_e1_srm, z_e2_srm, z_e3_srm, 1:10)
temp_90_10yr_avg_srm = calc_decadal_temp_avg(temp_exp_e1_srm, temp_exp_e2_srm, temp_exp_e3_srm,
                                             z_e1_srm, z_e2_srm, z_e3_srm, 61:71)


avg_temp_diff_nosrm = temp_90_10yr_avg_nosrm - temp_30_10yr_avg_nosrm
avg_temp_diff_srm = temp_90_10yr_avg_srm - temp_30_10yr_avg_srm
avg_temp_2100 =  temp_90_10yr_avg_srm - temp_90_10yr_avg_nosrm

# Diverging maps using GMT (centered at 0, symmetric linear scale)
make_diverging_linear_map_symmetric(avg_temp_diff_nosrm, "", "no_srm_tempdiff_linear.pdf")
make_diverging_linear_map_symmetric(avg_temp_diff_srm, "", "srm_tempdiff_linear.pdf")
make_diverging_linear_map_symmetric(avg_temp_2100, "", "tempdiff_2100_linear.pdf")

# Function to calculate population based 90-10 ratio
function pop_based_ratio(val1, val2, val3)

    val1 = hcat(1:19240, val1)
    val2 = hcat(1:19240, val2)
    val3 = hcat(1:19240, val3)

    e1_rat = zeros(71)
    e2_rat = zeros(71)
    e3_rat = zeros(71)
    rat = zeros(71)

    first_idx_A = Vector{Int32}(undef, 71)
    last_idx_A =  Vector{Int32}(undef, 71)
    first_idx_B = Vector{Int32}(undef, 71)
    last_idx_B = Vector{Int32}(undef, 71)
    first_idx_C = Vector{Int32}(undef, 71)
    last_idx_C = Vector{Int32}(undef, 71)

    first_idx_A_orig = zeros(71)
    last_idx_A_orig =  zeros(71)
    first_idx_B_orig = zeros(71)
    last_idx_B_orig = zeros(71)
    first_idx_C_orig = zeros(71)
    last_idx_C_orig = zeros(71)

    e1_10 = zeros(71)
    e1_90 = zeros(71)
    e2_10 = zeros(71)
    e2_90 = zeros(71)
    e3_10 = zeros(71)
    e3_90 = zeros(71)

    avg_10 = zeros(71)
    avg_90 = zeros(71)
    for i in 1:71

        A = [val1[:,i+1] popi[:,40 + i] val1[:,1]]
        A = A[sortperm(A[:,1]), :]
        y = cumsum(A[:,2])

        first_idx_A[i] = findfirst(>(glob_pop_10[40 + i]), y)
        last_idx_A[i] = findfirst(>(glob_pop_90[40 + i]), y)
        last_idx_A_orig[i] = A[last_idx_A[i], end]
        first_idx_A_orig[i] = A[first_idx_A[i], end]

        e1_rat[i] = A[last_idx_A[i], 1] / A[first_idx_A[i], 1]

        e1_90[i] = A[last_idx_A[i], 1]
        e1_10[i] = A[first_idx_A[i], 1]

        B = [val2[:,i+1] popi[:,40 +i ] val2[:,1]]
        B = B[sortperm(B[:,1]), :]

        first_idx_B[i] = findfirst(>(glob_pop_10[40 + i]), cumsum(B[:,2]))
        last_idx_B[i] = findfirst(>(glob_pop_90[40 + i]), cumsum(B[:,2]))
        last_idx_B_orig[i] = B[last_idx_B[i], end]
        first_idx_B_orig[i] = B[first_idx_B[i], end]

        e2_rat[i] = B[last_idx_B[i], 1] / B[first_idx_B[i], 1]

        e2_90[i] = B[last_idx_B[i], 1]
        e2_10[i] = B[first_idx_B[i], 1]

        C = [val3[:,i+1] popi[:,40 + i] val3[:,1]]
        C = C[sortperm(C[:,1]), :]
        first_idx_C[i] = findfirst(>(glob_pop_10[40 + i]), cumsum(C[:,2]))
        last_idx_C[i] = findfirst(>(glob_pop_90[40 + i]), cumsum(C[:,2]))
        last_idx_C_orig[i] = C[last_idx_C[i], end]
        first_idx_C_orig[i] = C[first_idx_C[i], end]

        e3_rat[i] = C[last_idx_C[i], 1] / C[first_idx_C[i], 1]

        e3_90[i] = C[last_idx_C[i], 1]
        e3_10[i] = C[first_idx_C[i], 1]

        rat[i] = (e1_rat[i] + e2_rat[i] + e3_rat[i])/3
        avg_10[i] = (e1_10[i] + e2_10[i] + e3_10[i])/3
        avg_90[i] = (e1_90[i] + e2_90[i] + e3_90[i])/3
    end

    return (rat = rat, e1rat = e1_rat, e2rat = e2_rat, e3rat = e3_rat, e1_10 = e1_10,
    e2_10 = e2_10, e3_10 = e3_10, e1_90 = e1_90, e2_90 = e2_90, e3_90 = e3_90, avg_10 = avg_10,
    avg_90 = avg_90, first_idx_A_orig, first_idx_B_orig, first_idx_C_orig , last_idx_A_orig,
    last_idx_B_orig, last_idx_C_orig)
end

nosrm_res = pop_based_ratio(gdpcap_e1_nosrm, gdpcap_e2_nosrm, gdpcap_e3_nosrm)
srm_res = pop_based_ratio(gdpcap_e1_srm, gdpcap_e2_srm, gdpcap_e3_srm)

# Plot population-based 90/10 ratios for all runs
years = 2030:2100
ratio_plot = Plots.plot(years, nosrm_res.e1rat, linewidth=1, color=:blue, alpha=0.5, label=false, title = "Population-based 90-10 Ratio for GDP/capita")
Plots.plot!(ratio_plot, years, nosrm_res.e2rat, linewidth=1, color=:blue, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, nosrm_res.e3rat, linewidth=1, color=:blue, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, nosrm_res.rat, linewidth=3, color=:blue, label="No SRM")
Plots.plot!(ratio_plot, years, srm_res.e1rat, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.e2rat, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.e3rat, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.rat, linewidth=3, color=:red, label="SRM")
Plots.xlabel!(ratio_plot, "Year")
Plots.ylabel!(ratio_plot, "90/10 Ratio")
savefig(ratio_plot, output_path * "pop_90_10_ratio_srm_nosrm.png")


# Decompose population 90/10 graph: 10
years = 2030:2100
ratio_plot = Plots.plot(years, nosrm_res.e1_10 * 1e6, linewidth=1, color=:blue, alpha=0.5, label=false, title = "10th percentile GDP/capita")
Plots.plot!(ratio_plot, years, nosrm_res.e2_10 * 1e6, linewidth=1, color=:blue, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, nosrm_res.e3_10 * 1e6, linewidth=1, color=:blue, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, nosrm_res.avg_10 * 1e6, linewidth=3, color=:blue, label="No SRM")
Plots.plot!(ratio_plot, years, srm_res.e1_10 * 1e6, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.e2_10 * 1e6, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.e3_10 * 1e6, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.avg_10 * 1e6, linewidth=3, color=:red, label="SRM")
Plots.xlabel!(ratio_plot, "Year")
Plots.ylabel!(ratio_plot, "GDP/capita (\$)")
savefig(ratio_plot, output_path * "pop_9010_ratio_decomp_10.png")


# Decompose population 90/10 graph: 90
years = 2030:2100
ratio_plot = Plots.plot(years,nosrm_res.e1_90 * 1e6, linewidth=1, color=:blue, alpha=0.5, label=false, title = "90th percentile GDP/capita")
Plots.plot!(ratio_plot, years, nosrm_res.e2_90 * 1e6, linewidth=1, color=:blue, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, nosrm_res.e3_90 * 1e6, linewidth=1, color=:blue, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, nosrm_res.avg_90 * 1e6, linewidth=3, color=:blue, label="No SRM")
Plots.plot!(ratio_plot, years, srm_res.e1_90 * 1e6, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.e2_90 * 1e6, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.e3_90 * 1e6, linewidth=1, color=:red, alpha=0.5, label=false)
Plots.plot!(ratio_plot, years, srm_res.avg_90 * 1e6, linewidth=3, color=:red, label="SRM")
Plots.xlabel!(ratio_plot, "Year")
Plots.ylabel!(ratio_plot, "GDP/capita (\$)")
savefig(ratio_plot, output_path * "pop_9010_ratio_decomp_90.png")

function calc_avg_gdp_pct_change(gdp_e1, gdp_e2, gdp_e3)
    gdpcap_e1_pct = zeros(ncells)
    gdpcap_e2_pct = zeros(ncells)
    gdpcap_e3_pct = zeros(ncells)
    for i in 1:ncells
        gdpcap_e1_pct[i] = (mean(gdp_e1[i, end-10:end]) - mean(gdp_e1[i, 1:10]))/mean(gdp_e1[i, 1:10]) * 100
        gdpcap_e2_pct[i] = (mean(gdp_e2[i, end-10:end]) - mean(gdp_e2[i, 1:10]))/mean(gdp_e2[i, 1:10]) * 100
        gdpcap_e3_pct[i] = (mean(gdp_e3[i, end-10:end]) - mean(gdp_e3[i, 1:10]))/mean(gdp_e3[i, 1:10]) * 100
    end
    avg_pct = (gdpcap_e1_pct + gdpcap_e2_pct + gdpcap_e3_pct) / 3
    return avg_pct
end

# Calculate average GDP percent change for no-SRM runs
nosrm_gdp_pct_change = calc_avg_gdp_pct_change(gdpcap_e1_nosrm, gdpcap_e2_nosrm, gdpcap_e3_nosrm)

# Calculate average GDP percent change for SRM runs
srm_gdp_pct_change = calc_avg_gdp_pct_change(gdpcap_e1_srm, gdpcap_e2_srm, gdpcap_e3_srm)

# Symmetric linear maps with fixed range -70 to 70 (same style as tempdiff_2100_linear.pdf)
make_symmetric_linear_map(nosrm_gdp_pct_change, 70, "nosrm_gdp_pct_change_sym70.pdf")
make_symmetric_linear_map(srm_gdp_pct_change, 70, "srm_gdp_pct_change_sym70.pdf")
make_symmetric_linear_map(nosrm_gdp_pct_change - srm_gdp_pct_change, 70, "gdp_pct_change_diff_sym70.pdf")
