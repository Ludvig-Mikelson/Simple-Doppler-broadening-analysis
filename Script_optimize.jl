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




names = glob("*.txt","C:/Users/Deloading/Desktop/Julia/LC/CsD2_Spektrs/")


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
function Min_Max(range_, up_down, dati, max_min)
    result = [] 
    
    if max_min == "min" 
        for i in range_
            if all(dati[i-d] > dati[i] && dati[i+d] > dati[i] for d in up_down)
                push!(result, dati[i])
            end
        end
        return result 
    elseif max_min == "max"  
        for i in range_
            if all(dati[i-d] < dati[i] && dati[i+d] < dati[i] for d in up_down)
                push!(result, dati[i])
            end
        end
        return result
    end      
end 

#Finds x value of y values
function x_finder(dati_points,dati)
    x_value = []
    for value in dati_points
        index = findfirst(x -> x == value, dati)
        push!(x_value,x[index])
    end  
    return x_value  
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

function FWHM(time_set,gauss_set,margin)
    max, index = findmax(gauss_set)


    gauss_set_first_half = gauss_set[1:index]
    gauss_set_second_half = gauss_set[index+1:end]
    println(index)

    index_FWHM_1 = findfirst(x -> abs(x - max/2) < margin, gauss_set_first_half)
    index_FWHM_2 = findfirst(x -> abs(x - max/2) < margin, gauss_set_second_half) + index
    index_FWHM = [index_FWHM_1,index_FWHM_2]

    println(index_FWHM)
    time_FWHM_1 = time_set[index]
    time_FWHM_2 = time_set[index_FWHM_2]
    if time_FWHM_1 > time_FWHM_2
        d_t = time_FWHM_1 - time_FWHM_2
    else
        d_t = time_FWHM_2 -time_FWHM_1
    end
    return d_t,index_FWHM
end

function temperaturez(index,f_trans, x_value)
    d_f = -time_to_frequency(x_value[index[1]]) + time_to_frequency(x_value[index[2]])
    temperature = (((d_f)^2) * (m) * ((c)^2)) / (((BigInt(f_trans)^2)) * k * 8 * (log(2)))
    return temperature
end

function normalizer(data)
    change = findmax(data)
    gauss__normalized = -data .+ change[1]
    return gauss__normalized
end

function FWHM_P(data_x,data_y,index)
    points = [data_y[index[1]], data_y[index[2]]]
    time = [data_x[index[1]], data_x[index[2]]]
    return time,points
end

function Gaus_3_least_square(x,y,guess,offset_set)
    fun(t,p) = p[1] .* exp.(-1 .* (t .- p[2]).^2 ./ (2 .* p[3].^2)) .+ p[4]
    names = []
    for i in eachindex(offset_set)
        push!(names, range(offset_set[i] - offset_set[i]/100, offset_set[i] + offset_set[i]/100, step = (offset_set[i]/(1e4))))
    end
    

    best_error = Inf
    fitted_3_peaks_coef = nothing
    offset =[]
    for i in range(1,(length(names[1])-1),step = 1)
        fun_3_peaks_l(t,p) = ((p[1] .* exp.(-1 .* (t .-names[1][i]).^2 ./ (2 .* p[2].^2)) .+ p[3]) + 
        (p[4] .* exp.(-1 .* (t .-names[2][i]).^2 ./ (2 .* p[5].^2)) .+ p[6]) +
        (p[7] .* exp.(-1 .* (t .-names[3][i]).^2 ./ (2 .* p[8].^2)) .+ p[9]))
    
        fitted_3_fit = LsqFit.curve_fit(fun_3_peaks_l,x,y,guess)
        error = sum(abs2,y .- fun_3_peaks_l(x,fitted_3_fit.param))
        if error < best_error
            best_error = error
            fitted_3_peaks_coef = fitted_3_fit.param
            push!(offset,names[1][i])
            push!(offset,names[2][i])
            push!(offset,names[3][i])
        end

    end
    #println(offset)
    #println(fitted_3_peaks_coef)
    fun_3_peaks(t,p) = ((p[1] .* exp.(-1 .* (t .-offset[1]).^2 ./ (2 .* p[2].^2)) .+ p[3]) + 
    (p[4] .* exp.(-1 .* (t .-offset[2]).^2 ./ (2 .* p[5].^2)) .+ p[6]) +
    (p[7] .* exp.(-1 .* (t .-offset[3]).^2 ./ (2 .* p[8].^2)) .+ p[9]))
   

    fitted_3_peaksz = fun_3_peaks(x,fitted_3_peaks_coef)

    gauss_1_coef = [fitted_3_peaks_coef[1],offset[1],fitted_3_peaks_coef[2],fitted_3_peaks_coef[3]]
    gauss_2_coef = [fitted_3_peaks_coef[4],offset[2],fitted_3_peaks_coef[5],fitted_3_peaks_coef[6]]
    gauss_3_coef = [fitted_3_peaks_coef[7],offset[3],fitted_3_peaks_coef[8],fitted_3_peaks_coef[9]]
    gaussz_1 = fun(x,gauss_1_coef) 
    gaussz_2 = fun(x,gauss_2_coef) 
    gaussz_3 = fun(x,gauss_3_coef) 
    #println(gaussz_1)
    return fitted_3_peaksz, gaussz_1, gaussz_2, gaussz_3
end



d_f_4_3 = 374038385.379628
d_f_4_4 = 374038626.659658
d_f_4_5 = 374038893.683144


C_3_4 = 351730902153880  #Hz
C_3_3 = 351731090642470  #Hz
C_3_2 = 351731040580070  #Hz

C_4_5 = 351721960613710  #Hz
C_4_4 = 351721709522110  #Hz
C_4_3 = 351721482637990  #Hz

offsets_f = [C_4_3,C_4_4,C_4_5]

m = 2.20694650 * 10^-25 #kg
c = 299792458 #m/s
k = 1.380649*10^-23 #m^2 kg s^-2 K^-1
f_0 = 351.72571850*10^12 #Hz

y_SAS = SAS_CsD2_fg4_column_2
y_Dop = Dop_CsD2_fg4_column_2 .-0.013
y_Fons = Fons_CsD2_fg4_column_2
x = SAS_CsD2_fg4_column_1
x_1 = SAS_CsD2_fg4_column_1
y_Sas_Fons = y_SAS - y_Fons
#y_final = broadcast(abs,-y_Dop + (y_SAS -y_Fons))

y_Sas_Fons_shifted = x_shifter(x,y_Sas_Fons,10)
y_Sas_Fons_shifted_final = y_Sas_Fons_shifted - y_Dop

minima = Min_Max(240:700,[1,2,3,4,5,6],y_Sas_Fons_shifted_final,"min")
time_minima = x_finder(minima,y_Sas_Fons_shifted_final)

minima = deleteat!(minima,[3])
time_minima = deleteat!(time_minima,[3])
coef_adjust = exp_fit(time_minima,minima)
adjust(a) = coef_adjust[1]*exp(coef_adjust[2]*a)
adjust_line = []

for i in 1:length(x)
    point_line = adjust(x[i])
    push!(adjust_line,point_line)
end
adjusted = y_Sas_Fons_shifted_final - adjust_line



maxima_adjusted = Min_Max(240:700,[1,2,3,4,5,6,7,8,9,10,11,12,13],adjusted,"max")
time_maxima_adjusted = x_finder(maxima_adjusted,adjusted)

d_t_1 = time_maxima_adjusted[4] - time_maxima_adjusted[6]
d_t_2 = time_maxima_adjusted[1] - time_maxima_adjusted[6]
proportion_t = d_t_1/d_t_2

d_f_1 = C_4_4 - C_4_5
d_f_2 = C_4_3 - C_4_5
proportion_f = d_f_1/d_f_2

ft_time = [time_maxima_adjusted[1], time_maxima_adjusted[4],time_maxima_adjusted[6]]
ft_frequency = [C_4_3,C_4_4,C_4_5]
coef_ft = CurveFit.linear_fit(ft_time,ft_frequency)
time_to_frequency(a) = coef_ft[1] .+ coef_ft[2]*a
frequency = time_to_frequency(x)
wavelength = float(299792458)./frequency*10^9
frequency_to_wavelength(b) = float(299792458)./b*10^9
time_to_wavelength(a) = frequency_to_wavelength(time_to_frequency(a))

data_mean = mean(y_Dop)
data_std = Statistics.std(y_Dop)
amplitude = maximum(y_Dop) - minimum(y_Dop)
minimum_index = argmin(y_Dop)
position_minimum = x[minimum_index]


fun(t,p) = p[1] .* exp.(-1 .* (t .- p[2]).^2 ./ (2 .* p[3].^2)) .+ p[4]
fun_1(t,p) = -amplitude .* exp.(-1 .* (t .- p[1]).^2 ./ (2 .* p[2].^2)) .+ p[3]
guess = [position_minimum,data_std,0.2]
fitted = LsqFit.curve_fit(fun_1,x_1,y_Dop,guess)
fitted_coef = fitted.param 
array = range(-1, stop=1, step = 0.001) |> collect
fitted_fianl = fun_1(array,fitted_coef)

guess_3_peaks = [amplitude ,data_std, 0.0001,
amplitude  ,data_std,0.0001,
amplitude ,data_std,0.0001]
y_dop_normalized = normalizer(y_Dop) 
pointstime = [x[(1120-300)],x[(1130-300)],x[(1150-300)],x[(1251-300)],x[(990)], x[(1000)]]
pointsy = [y_dop_normalized[(1120-300)],y_dop_normalized[(1130-300)],0,0,0,0]

fit_coefs = poly_fit(pointstime,pointsy,2)
fitted_curve(r,c) = c[1] .+ c[2] .* r .+ c[3] .* r .^2
offsets_t = [time_maxima_adjusted[1],time_maxima_adjusted[4],time_maxima_adjusted[6]]

padding = fill(0.0, 300)

x_extended = x
stepper = abs(x[2] - x[1])
min_val = minimum(x)
max_val = maximum(x)

x_extended = vcat(range(minimum(x), length=300, step=-stepper), x_extended, range(maximum(x), length=300, step=stepper))
fitted_end_time = x_extended[1120:1600]
fitted_end_y = fitted_curve(fitted_end_time,fit_coefs)
extended_array = vcat(padding, y_dop_normalized, padding)
extended_array[1120:1600] = fitted_curve(fitted_end_time,fit_coefs)
extended_array[1180:1600] .= 0
fitted_3_peaks_final, gauss_1,gauss_2,gauss_3 = Gaus_3_least_square(x_extended,extended_array,guess_3_peaks,offsets_t)

change = findmax(gauss_1)
gauss_1_normalized = -gauss_1 .+ change[1]
gauss_2_normalized = -gauss_2 .+ change[1]
gauss_3_normalized = -gauss_3 .+ change[1]
d_t_gauss_1, index_1 = FWHM(x_extended,gauss_1,0.0015)
d_t_gauss_2, index_2 = FWHM(x_extended,gauss_2,0.0015)
d_t_gauss_3, index_3 = FWHM(x_extended,gauss_3,0.0015)

gauss_1_time,gauss_1_points = FWHM_P(x_extended,gauss_1,index_1)
gauss_2_time,gauss_2_points = FWHM_P(x_extended,gauss_2,index_2)
gauss_3_time,gauss_3_points = FWHM_P(x_extended,gauss_3,index_3)

temp_4_3 = temperaturez(index_1,C_4_3,x_extended)
temp_4_4 = temperaturez(index_2,C_4_4,x_extended)
temp_4_5 = temperaturez(index_3,C_4_5,x_extended)

formatted_temp_4_3 = round(temp_4_3, sigdigits=3)
formatted_temp_4_4 = round(temp_4_4, sigdigits=3)
formatted_temp_4_5 = round(temp_4_5, sigdigits=3)

change_fitted_1 = findmax(fitted_fianl)
fitted_1_normalized = -fitted_fianl .+ change_fitted_1[1]

d_t_fitted_1, index_fitted_1 = FWHM(array,fitted_1_normalized,0.0008)
temp_central =  temperaturez(index_fitted_1,C_4_5,array)
temp_central_formated = round(temp_central, sigdigits=3)
fitted_1_points = [fitted_1_normalized[index_fitted_1[1]], fitted_1_normalized[index_fitted_1[2]]]
fitted_1_time = [array[index_fitted_1[1]], array[index_fitted_1[2]]]

change_dop = findmax(y_Dop)
dop_normalized = -y_Dop .+ change_dop[1]

new_set =[]
for i in 500:1000
    push!(new_set,x[i])
end

#d_t_dop, index_dop = FWHM(x,y_Dop,0.005)
#d_t_dop_2, index_dop_2 = FWHM(new_set,y_Dop,0.005)
#dop_time, dop_points = FWHM_P(x,y_Dop,index_dop)
#dop_time_2, dop_points_2 = FWHM_P(x,y_Dop,index_dop_2)
#temp_dop = temperaturez(index_dop,f_0,x)

#print(temp_central)
#print("\n")
#print(temp_4_3)
print("\n")
##print(temp_4_4)
print("\n")
#print(temp_4_5)

adjust_for_graph = 0.361


guess_3_peaks_f = [-amplitude ,data_std, 0.3,
-amplitude  ,data_std,
-amplitude ,data_std]

fitted_3_peaks_final_f, gauss_1_f,gauss_2_f,gauss_3_f = Gaus_3_least_square((frequency ./10^9),y_Dop,guess_3_peaks,(offsets_f ./10^9))


#titlez =  "GGGGGG$formatted_temp_4_3 $formatted_temp_4_4 $formatted_temp_4_5"

#plot(x,[y_Dop y_Sas_Fons_shifted y_Sas_Fons_shifted_final], label = ["Dop" "SAS" "SAS - Dop"], legend =:topright )

#plot(x,[adjusted y_Sas_Fons_shifted_final convert.(Float64,adjust_line)],
#ylims=(-0.005,0.11), label = ["Flattened with exp" "SAS - Dop" "exp"] )

#scatter!(time_maxima_adjusted,maxima_adjusted)

#plot(x_extended,
#[fitted_3_peaks_final (extended_array  )  ],
 #label = ["3 gauss fit" "Dop" "Fg4-3 gauss" "Fg4-4 gauss" "Fg4-5 gauss" ])
#plot!(gauss_1_time,gauss_1_points)
#plot!(gauss_2_time,gauss_2_points)
#plot!(gauss_3_time,gauss_3_points)
#scatter!(x_extended[[570, 738, 1120]], [0.1, 0.1, 0.1])

#plot(x_extended,extended_array)

#plot(wavelength, [gauss_1_normalized gauss_2_normalized gauss_3_normalized],
#label = ["Fg4-3" "Fg4-4" "Fg4-5"])
#plot!(time_to_wavelength(gauss_1_time),gauss_1_points, label = ["$formatted_temp_4_3 K"])
#plot!(time_to_wavelength(gauss_2_time),gauss_2_points, label = ["$formatted_temp_4_4 K"])
#plot!(time_to_wavelength(gauss_3_time),gauss_3_points, label = ["$formatted_temp_4_5 K"])

#plot(wavelength,[y_Dop] , label = "Dop")
#plot!(time_to_wavelength(array), fitted_fianl, label = "Fitted 1 gauss")

#plot(time_to_wavelength(array),fitted_1_normalized, label = "Fitted_normalized")
#plot!(time_to_wavelength(fitted_1_time),fitted_1_points, label = "$temp_central_formated K")

titlez = " gggggsdgsd3 Gaus fit, frequency"
plot(frequency./10^9,[fitted_3_peaks_final_f y_Dop (gauss_1_f .+ adjust_for_graph) (gauss_2_f .+ adjust_for_graph) (gauss_3_f .+ adjust_for_graph)],
label = ["3 gauss fit" "Dop" "Fg4-3 gauss" "Fg4-4 gauss" "Fg4-5 gauss"],
legend = :bottomleft)


#xlabel!("Time [s]")
xlabel!("frequency [Ghz]")
#xlabel!("wavelength [nm]")
ylabel!("Intensity [V]")
title!(titlez)
savefig(joinpath(@__DIR__, "Graphs_new", "$titlez.pdf"))