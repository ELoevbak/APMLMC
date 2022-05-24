import sys
sys.path.append("../../")
import numpy as np
import mlmc
import math

def level_timestep(level, epsilon, dt0, scale_factor, level_1_refinement):
    epsilon = epsilon/np.sqrt(level_1_refinement)
    ratio = dt0/epsilon**2
    if level==0:
        #Clean up any rounding errors
        return (epsilon**2 * round(ratio), 0)
    if level==1:
        return (epsilon**2, round(ratio))
    return (epsilon**2 / scale_factor**(level-1), scale_factor)

def formatValue(value, significantDigits):
    #Leave guard digit which we use later to aviod truncation
    formatString = "{:." + str(significantDigits-1) + "e}"
    formattedNumber = formatString.format(value)
    base, exponent = formattedNumber.split("e")
    sign = exponent[0]
    exponent = exponent.lstrip("+")
    exponent = exponent.lstrip("-")
    exponent = exponent.lstrip("0")
    if len(exponent) == 0:
        exponent = "0"
    if len(exponent) == 1:
        exponent += "\\phantom{0}"
    if sign == "-":
        exponent = sign + exponent
    
    phantomMinus = ""
    if exponent[0] != "-":
        phantomMinus = "\phantom{-}"
    result = "$" + base + " \\tabletimes 10^{" + exponent + phantomMinus + "}$"
    return result

def ensureMinusSpacing(formattedString):
    if formattedString[1] != "-":
        formattedString = formattedString[0] + "\phantom{-}" + formattedString[1:]
    return formattedString


data = mlmc.load_checkpoint(sys.argv[1])
epsilon = float(sys.argv[2])
dt0 = float(sys.argv[3])
scale_factor = int(sys.argv[4])
level_1_refinement = int(sys.argv[5])

#Print table header
print("\\begin{minipage}{\\textwidth}")
print("\\centering")
print("\\captionof{table}{$\\epsilon = " + str(epsilon) + "$, $\\Delta t_1 = " + formatValue(level_timestep(1, epsilon, dt0, scale_factor,level_1_refinement)[0],3)[1:-1].replace("tabletimes", "times") + "$}") #Cut away first dollar sign from formatValue result
print("\\centering ")
print("\\begin{tabular}{c | c c c c c | c}")
print("$\\ell$ & $\\Delta t_\\ell$ & $P_\\ell$ & \\multicolumn{1}{c}{$\\mathbb{E}[ \\hat{F}_\\ell \\! - \\! \\hat{F}_{\\ell-1} ]$} & $\\mathbb{V}[ \\hat{F}_\\ell \\! - \\! \\hat{F}_{\\ell-1} ]$ & $\\mathbb{V}[\\hat{Y}_\\ell]$ & $P_\\ell C_\\ell$ \\\\")
print("\\hline")

variance_sum = 0.0

#Print rows
for l in range(data.num_levels):
    timestep = level_timestep(l, epsilon, dt0, scale_factor, level_1_refinement)[0]
    samples = data.samples[l]
    diff_mean = data.diff_mean(l)[-1][-1]
    diff_variance = data.diff_variance(l,True)
    variance_sum += diff_variance/samples
    cost = data._cost[l] / 50.0 # Cost is rescaled as the simulation outputs cost in terms of simulations with time step size 0.5, but in publications simulations with time step size 0.01 are used as the reference
    print(l, " & ", formatValue(timestep,2), " & ", formatValue(samples, 2), " & ", ensureMinusSpacing(formatValue(diff_mean, 3)), " & ", formatValue(diff_variance, 2), " & ", formatValue(diff_variance/samples, 2), " & ", formatValue(samples*cost,2), " \\\\")

#Print summary data
print("\\hline")
print("$\\sum$ & \\multicolumn{1}{c}{} & & " + ensureMinusSpacing(formatValue(data.telescopic_sum()[-1][-1][-1], 3)) + " & & " + formatValue(variance_sum, 2) + " & " + formatValue(data.total_work/50.0, 2)) # See comment above for factor 50

#Print table termination
print("\\end{tabular}")
print("\\vspace{15pt}")
print("\\end{minipage}")