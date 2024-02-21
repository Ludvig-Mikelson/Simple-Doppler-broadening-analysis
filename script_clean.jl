using CSV
using DataFrames
using Plots
using Glob
using Polynomials
using Distributions
using Statistics
using LsqFit
using CurveFit
using LinearAlgebra
using Polynomials
using NonlinearSolve
using Printf
using LaTeXStrings
using DifferentialEquations
using SpecialFunctions


namez = glob("*.txt",@__DIR__)

#Loads data and gives appropriate names
for file_path in namez
    global data1 = DataFrame(CSV.File(file_path))

    file_name = splitdir(file_path)[2]
    file_name = splitext(file_name)[1]

    set_name_column1 = Symbol("$(file_name)_column_1")
    set_name_column2 = Symbol("$(file_name)_column_2")

    eval(quote
    $set_name_column1 = data1[:, 1]
    $set_name_column2 = data1[:, 2]
    end)
end

# Finds min or max points
function Extrema(range_x, range_y, x, y, maxORmin)
    result = [] 
    time = []
    
    if maxORmin == "min" 
        for i in range_x
            if all(y[i-d] > y[i] && y[i+d] > y[i] for d in range_y)
                push!(result, y[i])
                push!(time, x[i])
            end
        end
        return result 
    elseif maxORmin == "max"  
        for i in range_x
            if all(y[i-d] < y[i] && y[i+d] < y[i] for d in range_y)
                push!(result, y[i])
                push!(time, x[i])
            end
        end
        return time, result
    end      
end 


#Shifts data to left or right by shift*(x step)
function x_shifter(x_data, y_data, shift)
    y_data_shifted = zeros(eltype(y_data), length(y_data))
    for i in (abs(shift) + 1):(length(x_data) - abs(shift))
        y_data_shifted[i] = y_data[i + shift]
    end

    for i in 1:shift
        if y_data_shifted[i] ≈ 0
            y_data_shifted[i] = y_data_shifted[2*abs(shift)]
        end
    end
    for i in (length(y_data)-abs(shift)):length(y_data)
        if y_data_shifted[i] ≈ 0
            y_data_shifted[i] = y_data_shifted[length(y_data)-abs(shift)]
        end
    end
    return y_data_shifted
end

#Finds FWHM and the associated coefficients
function FWHM(time_set,gauss_set,margin,cov_matrix)
    gauss_set = gauss_set .- findmin(gauss_set)[1]
    max, index = findmax(gauss_set)
    maxup = cov_matrix[2,2] + max
    maxdown = -cov_matrix[2,2] + max
    #println(maxup)
    #println("here")
    #println(maxdown)
    gauss_set_first_half = gauss_set[1:index]
    gauss_set_second_half = gauss_set[index+1:end]

    index_FWHM_1 = findfirst(x -> abs(x - max/2) < margin, gauss_set_first_half)
    index_FWHM_2 = findfirst(x -> abs(x - max/2) < margin, gauss_set_second_half) + index

    index_FWHM_1up = findfirst(x -> abs(x - maxup/2) < margin, gauss_set_first_half)
    index_FWHM_2up = findfirst(x -> abs(x - maxup/2) < margin, gauss_set_second_half) + index

    index_FWHM_1down = findfirst(x -> abs(x - maxdown/2) < margin, gauss_set_first_half)
    index_FWHM_2down = findfirst(x -> abs(x - maxdown/2) < margin, gauss_set_second_half) + index
    index_FWHM = [index_FWHM_1,index_FWHM_2]
    index_FWHMup = [(index_FWHM_1up),index_FWHM_2up]
    index_FWHMdown = [index_FWHM_1down,index_FWHM_2down]


    d_t_variance = (time_set[index_FWHM_2down] - time_set[index_FWHM_1down]) - (time_set[index_FWHM_2up] - time_set[index_FWHM_1up]) + 2*cov_matrix[1,1]
    #println(d_t_variance)
    #println("anotherone")
    time_FWHM_1 = time_set[index]
    time_FWHM_2 = time_set[index_FWHM_2]
    if time_FWHM_1 > time_FWHM_2
        d_t = time_FWHM_1 - time_FWHM_2
    else
        d_t = time_FWHM_2 -time_FWHM_1
    end
    return d_t,index_FWHM, d_t_variance, index_FWHMup, index_FWHMdown
end

#Given input in Hz finds temperarurez associated with FWHM
function temperarurez(index,f_trans, x_value, indexup, indexdown, cov)
    d_f = abs(abs(x_value[index[1]]) - abs(x_value[index[2]]))
    d_index_cov = cov/0.0005
    temperarurez = (((d_f)^2) * (m) * ((c)^2)) / (((BigInt(f_trans)^2)) * k * 8 * (log(2)))
    d_f_varianceup = abs(abs(x_value[indexup[1] + trunc(Int,d_index_cov)]) - abs(x_value[indexup[2] - trunc(Int,d_index_cov)]))
    d_f_variancedown = abs(abs(x_value[indexdown[1] - trunc(Int,d_index_cov)]) - abs(x_value[indexdown[2] + trunc(Int,d_index_cov)]))

    temperaturez_up = (((d_f_varianceup)^2) * (m) * ((c)^2)) / (((BigInt(f_trans)^2)) * k * 8 * (log(2)))
    temperaturez_down = (((d_f_variancedown)^2) * (m) * ((c)^2)) / (((BigInt(f_trans)^2)) * k * 8 * (log(2)))

    temp_var_up = temperaturez_up - temperarurez
    temp_var_down = temperaturez_down - temperarurez
    println(temp_var_up)
    println("up")
    println(temperarurez)
    println("down")
    println(temp_var_down)
    
    return temperarurez, temp_var_up, temp_var_down
end

function FlipToZero(data)
    change = findmax(data)
    gauss = -data .+ change[1]
    return gauss
end

#FWHM points for ploting
function FWHM_P(data_x,data_y,index)
    points = [data_y[index[1]], data_y[index[2]]]
    time = [data_x[index[1]], data_x[index[2]]]
    return time,points
end

function Gauss_3_least_square(x,y,offset_set)
    fun(t,p) = p[1] .* exp.(-1 .* (t .- p[2]).^2 ./ (2 .* p[3].^2)) .+ p[4]

    range_offset = []
    for i in eachindex(offset_set)
        if offset_set[i] > 0
            push!(range_offset, range(offset_set[i] - offset_set[i]/100, offset_set[i] + offset_set[i]/100, step = (offset_set[i]/(1e3))))
        elseif offset_set[i] < 0
            push!(range_offset, range(offset_set[i] + offset_set[i]/100, offset_set[i] - offset_set[i]/100, step = (abs(offset_set[i]/(1e3)))))
        end
    end
    
    std = Statistics.std(y)
    amplitude = maximum(y) - minimum(y)
    guess = [0,
             0,
             0]

    best_error, offset_best_1, offset_best_2, offset_best_3 = Inf,Inf,Inf,Inf
    fitted_3_peaks_coef = nothing
    std_locked = std/3.305
    amplitude_locked = amplitude/1.5


    for i in range(1,(length(range_offset[1])-1),step = 1)
            fun_3_peaks_restricted(t,p) =  (((amplitude*(7/72)) .* exp.(-1 .* (t .-range_offset[1][i]).^2 ./ (2 .* std_locked.^2)) .+ p[1]) + 
                                           ((amplitude*(7/24)) .* exp.(-1 .* (t .-range_offset[2][i]).^2 ./ (2 .* std_locked.^2)) .+ p[2]) +
                                           ((amplitude*(11/18)) .* exp.(-1 .* (t .-range_offset[3][i]).^2 ./ (2 .* std_locked.^2)) .+ p[3]))

            function problemz(p,t)
                     (((amplitude*(7/72)) .* exp.(-1 .* (t .-range_offset[1][i]).^2 ./ (2 .* std_locked.^2)) .+ p[1]) + 
                      ((amplitude*(7/24)) .* exp.(-1 .* (t .-range_offset[2][i]).^2 ./ (2 .* std_locked.^2)) .+ p[2]) +
                      ((amplitude*(11/18)) .* exp.(-1 .* (t .-range_offset[3][i]).^2 ./ (2 .* std_locked.^2)) .+ p[3])) .- y
            end

            probleme = NonlinearProblem(problemz,guess,x) 
            sol = solve(probleme,LevenbergMarquardt()) 
            error = sum(abs2,y .- fun_3_peaks_restricted(x,sol))

            if error < best_error
                best_error = error
                fitted_3_peaks_coef = sol
                offset_best_1 = range_offset[1][i]
                offset_best_2 = range_offset[2][i]
                offset_best_3 = range_offset[3][i] 
                mean_y = mean(y)
                mean_squares = sum((y .- mean_y).^2)
                residual = problemz(sol,x)
                residual_squares = sum(residual.^2)
                global r_square = 1 - (residual_squares/mean_squares)
                global chi_square = sum((residual .^2) / y)
                #println(r_square)
                #println(chi_square)
                
            end
        end

    
    fun_3_peaks(t,p) = (((amplitude*(7/72)) .* exp.(-1 .* (t .-offset_best_1).^2 ./ (2 .* std_locked.^2)) .+ p[1]) + 
                        ((amplitude*(7/24)) .* exp.(-1 .* (t .-offset_best_2).^2 ./ (2 .* std_locked.^2)) .+ p[2]) +
                        ((amplitude*(11/18)) .* exp.(-1 .* (t .-offset_best_3).^2 ./ (2 .* std_locked.^2)) .+ p[3]))

    covariance = Statistics.cov(hcat(fun_3_peaks(x, fitted_3_peaks_coef), y))


    extended = range(-0.5,0.5, step = 0.0005)                    
    fitted_3_peaksz = fun_3_peaks(extended,fitted_3_peaks_coef)

    gauss_1_coef = [(amplitude*(7/72)),offset_best_1,std_locked,fitted_3_peaks_coef[1]]
    gauss_2_coef = [(amplitude*(7/24)),offset_best_2,std_locked,fitted_3_peaks_coef[2]]
    gauss_3_coef = [(amplitude*(11/18)),offset_best_3,std_locked,fitted_3_peaks_coef[3]]
    print(fitted_3_peaks_coef)
    gaussz_1 = fun(extended,gauss_1_coef)
    gaussz_2 = fun(extended,gauss_2_coef) 
    gaussz_3 = fun(extended,gauss_3_coef) 

    return fitted_3_peaksz, gaussz_1, gaussz_2, gaussz_3, covariance
end


function voigt3_fit(xdata,ydata,offsets)
    std = Statistics.std(ydata)
    std_locked = std/3.305

    gaussian(x, p) = p[1] * exp.(-(x .- p[2]).^2 ./ (2 .* p[3].^2))
    cauchy(x, p) = p[1] ./ (π .* p[3] .* (1 .+ ((x .- p[2]) ./ p[3]).^2))

    function voigt(x, p)
        gauss_params = p[1:3]
        cauchy_params = p[4:6]
        return gaussian(x, gauss_params) + cauchy(x, cauchy_params)
    end

    function voigt3(x, p)
        p1 = [p[1],offsets[1],p[2],p[3],p[4],p[5]]
        p2 = [p[6],offsets[2],p[7],p[8],p[9],p[10]]
        p3 = [p[11],offsets[3],p[12],p[13],p[14],p[15]]
        return voigt(x, p1) + voigt(x, p2) + voigt(x, p3)
    end
    

    p0 = [2.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5,
    1.0,1.0,1.0,1.0,1.0,1.0]
    fit = LsqFit.curve_fit(voigt3, xdata, ydata, p0)
    best_params = coef(fit)

    coefs43 = [best_params[1],offsets[1],best_params[2]]
    coefs44 = [best_params[6],offsets[2],best_params[7]]
    coefs45 = [best_params[11],offsets[3],best_params[12]]

    fitted = voigt3(xdata,best_params)
    display(plot(xdata,[ydata fitted]))
    display(plot!(xdata,gaussian(xdata,coefs43)))
    display(plot!(xdata,gaussian(xdata,coefs44)))
    display(plot!(xdata,gaussian(xdata,coefs45)))
    return best_params
end

function Analysis(time, Dop, SAS)

    SAS_FlippedToZero = FlipToZero(SAS)
    Dop_FlippedToZero = FlipToZero(Dop)
    SAS_shifted = x_shifter(time,SAS_FlippedToZero,10)
    fit_SAS_coef = CurveFit.linear_fit([time[1], time[1000]],[Dop_FlippedToZero[1], Dop_FlippedToZero[1000]])
    line(a) = fit_SAS_coef[1] .+ fit_SAS_coef[2] .*a

    Dop_FlippedToZero = Dop_FlippedToZero - line(time)
    SAS_shifted = SAS_shifted -line(time)

    data_forPeaks = Dop_FlippedToZero - SAS_shifted
    time_Peaks, Peaks = Extrema(240:700,[1,2,3,4,5,6,7,8,9,10,11,12,13,14],time, data_forPeaks,"max")
    
    time_C4_3 = time_Peaks[1]
    time_C4_4 = time_Peaks[4]
    time_C4_5 = time_Peaks[6]


    Trans_t = [time_C4_3, time_C4_4, time_C4_5]
    Trans_f = [C_4_3, C_4_4 ,C_4_5] ./1e9 #GHz
    timeToFrequency_coef = CurveFit.linear_fit(Trans_t, Trans_f)
    time_to_frequency(a) = timeToFrequency_coef[1] .+ timeToFrequency_coef[2] .* a
    frequency = time_to_frequency(time) #GHz 
    extended = range(-0.5,0.5, step = 0.0005)
    extended_f = time_to_frequency(extended)


    

    Fit_full, Fit_4_3, Fit_4_4, Fit_4_5, covariance = Gauss_3_least_square(time,Dop_FlippedToZero, Trans_t)

    d_f_4_3, index4_3, variance4_3, indexup43, indexdown43 = FWHM(extended_f,Fit_4_3,0.001,covariance)
    d_f_4_4, index4_4, variance4_4, indexup44, indexdown44 = FWHM(extended_f,Fit_4_4,0.001,covariance)
    d_f_4_5, index4_5, variance4_5, indexup45, indexdown45 = FWHM(extended_f,Fit_4_5,0.001,covariance)
    d_f_dop, indexdop, variancedop, indexupdop, indexdowndop = FWHM(extended_f,Dop_FlippedToZero,0.001,covariance)
    timeset, pointset = FWHM_P(extended,Fit_4_3,index4_3)
    timeset1, pointset1 = FWHM_P(extended,Fit_4_5,index4_5)
    temp_4_3, temp_4_3up, temp_4_3down = temperarurez(index4_3,C_4_3,(extended_f .*10^9),indexup43, indexdown43,covariance[1,1])
    temp_4_4, temp_4_4up, temp_4_4down = temperarurez(index4_4,C_4_4,(extended_f .*10^9),indexup44, indexdown44,covariance[1,1])
    temp_4_5, temp_4_5up, temp_4_5down = temperarurez(index4_5,C_4_5,(extended_f .*10^9), indexup45, indexdown45,covariance[1,1])
    tempset = [temp_4_3up, temp_4_3down,temp_4_4up, temp_4_4down,temp_4_5up, temp_4_5down]
    #temp_dop = temperarurez(indexdop,f_0,(frequency .*10^9), indexupdop, indexdowndop,covariance[1,1])
    #println(temp_4_3)
    #println(temp_4_4)
    #println(temp_4_5)
    #println(temp_dop)
    #display(scatter(timeset,pointset))
    #display(scatter!(timeset1,pointset1))
    #display(plot!(extended,[Fit_4_3 Fit_4_4 Fit_4_5]))
    #display(plot!(time,Dop_FlippedToZero))

    voigt_coef = voigt3_fit(time,dopper,Trans_t)
    
    return Fit_full, Fit_4_3, Fit_4_4, Fit_4_5, Dop_FlippedToZero, frequency, SAS_FlippedToZero, extended_f, covariance,tempset

end

d_f_3_4 = 374048402.568479
d_f_3_3 = 374048603.016763
d_f_3_5 = 374048549.777879

d_f_4_3 = 374038385.379628 #Hz
d_f_4_4 = 374038626.659658 #Hz
d_f_4_5 = 374038893.683144 #Hz

C_3_4 = 351730902153880  #Hz
C_3_3 = 351731090642470  #Hz
C_3_2 = 351731040580070  #Hz

C_4_5 = 351721960613710  #Hz
C_4_4 = 351721709522110  #Hz
C_4_3 = 351721482637990  #Hz

m = 2.20694650 * 10^-25 #kg
c = 299792458 #m/s
k = 1.380649*10^-23 #m^2 kg s^-2 K^-1
f_0 = 351.72571850*10^12 #Hz

y_SAS = SAS_CsD2_fg4_column_2
y_Dop = Dop_CsD2_fg4_column_2 .-0.013
y_Fons = Fons_CsD2_fg4_column_2
time = SAS_CsD2_fg4_column_1
y_Sas_Fons = y_SAS - y_Fons


total, g_1,g_2,g_3, dopper, frequency,SAS, extended_f, covv, tempset = Analysis(time,y_Dop,y_Sas_Fons)


uncertanty = sqrt(covv[1,1])
tempset .= abs.(tempset)
tempset = [@sprintf("%.2f", x) for x in tempset]


plot(extended_f,[g_1],labels = ["Fg4-3, \$T = 289.18 K \frac{+$(tempset[2]) K}{-$(tempset[1]) K}\$"])
plot!(extended_f,[g_2],labels = ["Fg4-4, \$T = 291.28 K \frac{+$(tempset[4]) K}{-$(tempset[3]) K}\$"])
plot!(extended_f,[g_3],labels = ["Fg4-5, \$T = 291.28 K \frac{+$(tempset[6]) K}{-$(tempset[5]) K}\$"])
plot!(frequency,dopper, labels = false)
plot!(extended_f,total, labels = false)
plot!(legend=:outertop, legendcolumns=3)
annotate!([351721.0],[0.4],text("r = $(@sprintf("%.3f", r_square))"))
annotate!([351721.0],[0.35],text("chi = $(@sprintf("%.3f", chi_square))"))

plot!(size =(900,600))
titlez =  "3 Gauss Fit, locked position amplitude and position test"
ylabel!("Intensity [V]")
xlabel!("frequency [GHz]")
title!(titlez)


savefig(joinpath(@__DIR__, "Graphs_test", "$titlez.pdf"))

save_directory = joinpath(@__DIR__, "Graphs_test")

#var_names = Base.names(Main)
#var_values = [Main.eval(Symbol(var_name)) for var_name in var_names]
#variables_dict = Dict(zip(var_names, var_values))

#if !isdir(save_directory)
   # mkdir(save_directory)
#end

#open(joinpath(save_directory, "$titlez.txt"), "w") do file
 #   for (name, value) in variables_dict
  #      println(file, "$name = $value")
   # end
#end