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


names = glob("*.txt",@__DIR__)

#Loads data and gives appropriate names
for file_path in names
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
function FWHM(time_set,gauss_set,margin)
    max, index = findmax(gauss_set)

    gauss_set_first_half = gauss_set[1:index]
    gauss_set_second_half = gauss_set[index+1:end]

    index_FWHM_1 = findfirst(x -> abs(x - max/2) < margin, gauss_set_first_half)
    index_FWHM_2 = findfirst(x -> abs(x - max/2) < margin, gauss_set_second_half) + index
    index_FWHM = [index_FWHM_1,index_FWHM_2]

    time_FWHM_1 = time_set[index]
    time_FWHM_2 = time_set[index_FWHM_2]
    if time_FWHM_1 > time_FWHM_2
        d_t = time_FWHM_1 - time_FWHM_2
    else
        d_t = time_FWHM_2 -time_FWHM_1
    end
    return d_t,index_FWHM
end

#Given input in Hz finds temperature associated with FWHM
function temperature(index,f_trans, x_value)
    d_f = abs(abs(x_value[index[1]]) - abs(x_value[index[2]]))
    temperature = (((d_f)^2) * (m) * ((c)^2)) / (((BigInt(f_trans)^2)) * k * 8 * (log(2)))
    return temperature
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

    guess = [amplitude/2 ,std/3, 0,
             amplitude/2 ,std/3,
             amplitude/2 ,std/3]

    best_error = Inf
    fitted_3_peaks_coef = nothing
    offset_best_1 = Inf
    offset_best_2 = Inf
    offset_best_3 = Inf

    
    for i in range(1,(length(range_offset[1])-1),step = 1)
            fun_3_peaks_restricted(t,p) = ((p[1] .* exp.(-1 .* (t .-range_offset[1][i]).^2 ./ (2 .* p[2].^2)) .+ p[3]) .+ 
                                           (p[4] .* exp.(-1 .* (t .-range_offset[2][i]).^2 ./ (2 .* p[5].^2)) .+ p[3]) .+
                                           (p[6] .* exp.(-1 .* (t .-range_offset[3][i]).^2 ./ (2 .* p[7].^2)) .+ p[3]))
            fitted_3_fit = LsqFit.curve_fit(fun_3_peaks_restricted,x,y,guess)
            error = sum(abs2,y .- fun_3_peaks_restricted(x,fitted_3_fit.param))
            if error < best_error
                best_error = error
                fitted_3_peaks_coef = fitted_3_fit.param
                offset_best_1 = range_offset[1][i]
                offset_best_2 = range_offset[2][i]
                offset_best_3 = range_offset[3][i]

                
            end
        end

   
    fun_3_peaks(t,p) = ((p[1] .* exp.(-1 .* (t .-offset_best_1).^2 ./ (2 .* p[2].^2)) .+ p[3]) + 
                        (p[4] .* exp.(-1 .* (t .-offset_best_2).^2 ./ (2 .* p[5].^2)) .+ p[3]) +
                        (p[6] .* exp.(-1 .* (t .-offset_best_3).^2 ./ (2 .* p[7].^2)) .+ p[3]))
                        
    fitted_3_peaksz = fun_3_peaks(x,fitted_3_peaks_coef)
    println(fitted_3_peaks_coef[2])
    println(std)
    gauss_1_coef = [fitted_3_peaks_coef[1],offset_best_1,fitted_3_peaks_coef[2],fitted_3_peaks_coef[3]]
    gauss_2_coef = [fitted_3_peaks_coef[4],offset_best_2,fitted_3_peaks_coef[5],fitted_3_peaks_coef[3]]
    gauss_3_coef = [fitted_3_peaks_coef[6],offset_best_3,fitted_3_peaks_coef[7],fitted_3_peaks_coef[3]]
    gaussz_1 = fun(x,gauss_1_coef) 
    gaussz_2 = fun(x,gauss_2_coef) 
    gaussz_3 = fun(x,gauss_3_coef) 

    return fitted_3_peaksz, gaussz_1, gaussz_2, gaussz_3
end


function Analysis(time, Dop, SAS)

    SAS_shifted = x_shifter(time,SAS,10)
    data_forPeaks = Dop - SAS_shifted
    time_Peaks, Peaks = Extrema(240:700,[1,2,3,4,5,6,7,8,9,10,11,12,13],time, data_forPeaks,"max")
    
    time_C4_3 = time_Peaks[1]
    time_C4_4 = time_Peaks[4]
    time_C4_5 = time_Peaks[6]

    Trans_t = [time_C4_3, time_C4_4, time_C4_5]
    Trans_f = [C_4_3, C_4_4 ,C_4_5] ./1e9 #GHz
    timeToFrequency_coef = CurveFit.linear_fit(Trans_t, Trans_f)
    time_to_frequency(a) = timeToFrequency_coef[1] .+ timeToFrequency_coef[2] .* a
    frequency = time_to_frequency(time) #GHz
    Dop_FlippedToZero = FlipToZero(Dop)
    
    println(Trans_f)
    Fit_full, Fit_4_3, Fit_4_4, Fit_4_5 = Gauss_3_least_square(time,Dop_FlippedToZero, Trans_t)

    d_f_4_3, index4_3 = FWHM(frequency,Fit_4_3,0.002)
    d_f_4_4, index4_4 = FWHM(frequency,Fit_4_4,0.002)
    d_f_4_5, index4_5 = FWHM(frequency,Fit_4_5,0.002)

    temp_4_3 = temperature(index4_3,C_4_3,(frequency .*10^9))
    temp_4_4 = temperature(index4_4,C_4_4,(frequency .*10^9))
    temp_4_5 = temperature(index4_5,C_4_5,(frequency .*10^9))

    println(temp_4_3)
    println(temp_4_4)
    println(temp_4_5)

    return Fit_full, Fit_4_3, Fit_4_4, Fit_4_5, Dop_FlippedToZero, frequency

end



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

total, g_1,g_2,g_3, dopper, frequency = Analysis(time,y_Dop,y_Sas_Fons)

plot(frequency,[dopper total g_1 g_2 g_3],
labels = ["Doppler" "Fit" "Fg4-3, T = 257.6K" "Fg4-4, T = 612.5K" "Fg4-5, T = 379.5K"],
legend = :topleft)
plot!(size =(800,600))
titlez =  "Gauss Fit"
ylabel!("Intensity [V]")
xlabel!("frequency [GHz]")
title!(titlez)
savefig(joinpath(@__DIR__, "Graphs_test", "$titlez.pdf"))