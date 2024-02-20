using CSV
using DataFrames
using Plots
using Glob
using CurveFit
using Polynomials
using Distributions
using Statistics

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
function Min_Max(range_, up_down, dati, name, max_min)
    for i in range_
        if max_min == ">"
            if all(dati[i-d] > dati[i] && dati[i+d] > dati[i] for d in up_down)
                push!(name,dati[i])
            end
     
        elseif max_min == "<"
            if all(dati[i-d] < dati[i] && dati[i+d] < dati[i] for d in up_down)
                push!(name,dati[i])
            end
        end      
    end
    return name
end    


y_SAS = SAS_CsD2_fg4_column_2
y_Dop = Dop_CsD2_fg4_column_2 .-0.013
y_Fons = Fons_CsD2_fg4_column_2
x = SAS_CsD2_fg4_column_1
y_final = broadcast(abs,(-y_Dop + (y_SAS -y_Fons)))

Min_Max([240:700],[1,2],y_final,"minima",>)

#print(minima)

time_minima = []
for value in minima
    index = findfirst(x -> x == value, y_final)
    push!(time_minima,x[index])
end

print(time_minima)

minima = deleteat!(minima,[5])
time_minima = deleteat!(time_minima,[5])
coef_adjust = exp_fit(time_minima,minima)
adjust(a) = coef_adjust[1]*exp(coef_adjust[2]*a)
adjust_line = []

for value in x
    point_line = adjust(value)
    push!(adjust_line,point_line)
end
adjusted = y_final - adjust_line


C_3_4 = 351730902153880  
C_3_3 = 351731090642470
C_3_2 = 351731040580070

C_4_5 = 351721960613710.62
C_4_4 = 351721709522110.62
C_4_3 = 351721482637990.62
maxima = []
Min_Max([240:700], [1,2,3,4,5,6,7,8,9,10,11,12,13,14], adjusted, "maxima", <)
print(maxima)

maxima_adjusted = []
for i in 240:700
    if all(adjusted[i-d] < adjusted[i] && adjusted[i+d] < adjusted[i] for d in [1,2,3,4,5,6,7,8,9,10,11,12,13,14])

        push!(maxima_adjusted,adjusted[i])
    end
end

print(maxima_adjusted)

time_maxima_adjusted = []
for value in maxima_adjusted
    index = findfirst(x -> x == value, adjusted)
    push!(time_maxima_adjusted,x[index])
end

#print(time_maxima_adjusted)

d_t_1 = time_maxima_adjusted[4] - time_maxima_adjusted[6]
d_t_2 = time_maxima_adjusted[1] - time_maxima_adjusted[6]
proportion_t = d_t_1/d_t_2

d_f_1 = C_4_4 - C_4_5
d_f_2 = C_4_3 - C_4_5
proportion_f = d_f_1/d_f_2

ft_time = [time_maxima_adjusted[1], time_maxima_adjusted[4],time_maxima_adjusted[6]]
ft_frequency = [C_4_3,C_4_4,C_4_5]
coef_ft = linear_fit(ft_time,ft_frequency)
time_to_frequency(a) = coef_ft[1] .+ coef_ft[2]*a
frequency = time_to_frequency(x)
wavelength = float(299792458)./frequency*10^9
#print(proportion_t)
#print("\t")
#print(proportion_f)
#print(proportion_f/proportion_t)

data_mean = mean(y_Dop)
data_std = Statistics.std(y_Dop)
amplitude = maximum(y_Dop) - minimum(y_Dop)
minimum_index = argmin(y_Dop)
position_minimum = x[minimum_index]

normal_dist(a) = -amplitude * exp.(-(a .- position_minimum ).^2 / (2 * data_std^2)) .+ 0.5

array = range(-1, stop=1, step = 0.001) |> collect
array_1 = normal_dist(array)
index_FWHM = findall(x -> abs(x - amplitude/2) < 0.0012, array_1)
print(index_FWHM)
#print(amplitude)
#print(array_1[826])
time_FWHM_1 = array[index_FWHM[1]]
time_FWHM_2 = array[index_FWHM[2]]

m = 2.20694650 * 10^-25
c = 299792458
d_f = time_to_frequency(array[1261]) - time_to_frequency(array[826])
k = 1.380649*10^-23
f_0 = c/(852.3526*10^-9)
f_0_test = 351.72571850*10^12
temperature = (m*c^2*d_f^2)/((f_0_test^2)*8*k*log(2))

print(d_f)
print("\t")
print(temperature)
print("\t")
wavelength_test = (c/time_to_frequency(array[1261]) - c/time_to_frequency(array[826]))*10^9
print(wavelength_test)

#plot(x,[y_SAS y_Dop y_final convert.(Float64,adjust_line)], label = ["SAS" "Dop" "SAS - Dop"], legend= false)
#plot(x,[y_final convert.(Float64,adjust_line)], label = ["SAS - Fons" "Dop" "SAS - Dop - Fons"])
#plot(x, adjusted, ylims=(-0.01,0.11))
plot(x,[ y_Dop], xlims = (-1,1))
plot!(array,normal_dist(array))
scatter!([array[826]], [array_1[1261]] )
scatter!([array[1261]], [array_1[1261]] )


#scatter!(time_maxima_adjusted,maxima_adjusted)
xlabel!("wavelength [nm]")
ylabel!("V [100 mV]")
title!("Dop - 0.013, Raw-Data, SAS - Fons")
savefig(joinpath(@__DIR__, "Graphs", "graph1.pdf"))

