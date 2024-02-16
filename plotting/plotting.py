"""Subspace-Net plotting script 
    Details
    ------------
    Name: plotting.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 01/04/23

    Purpose
    ------------
    This script generates the plots which presented in the following papers: SubspaceNet journal paper:
    [1] D. H. Shmuel, J. Merkofer, G. Revach, R. J. G. van Sloun, and N. Shlezinger “Deep Root MUSIC Algorithm for Data-Driven DoA Estimation”, IEEE ICASSP 2023 
    [2] "SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation"
    
    The script uses the following functions:
        * create_dataset: For creating training and testing datasets 
        * run_simulation: For training DR-MUSIC model
        * evaluate_model: For evaluating subspace hybrid models

    This script requires that plot_style.txt will be included in the working repository.

"""
###############
#   Imports   #
###############
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

######################
# Conversion methods #
######################

def unit(value):
    return value

def rad2deg(value):
    return value * 180 / np.pi

def deg2rad(value):
    return value * np.pi / 180

def rad2dB(value:float):
    """Converts MSE value in radian scale into dB scale

    Args:
        mse_value (float): value of error in rad

    Returns:
        float: value of error in dB
    """    
    return 10 * np.log10(value)


def scenario_to_plot(simulation, conv_method=unit, mode = "non-coherent", T = 200):
    Loss = {}
    x_axis = {}
    
    # Criterion
    RMSPE = True
    if conv_method == rad2dB:
        RMSPE = False

    ###### mis-calibration simulations  ######
    if simulation.startswith("distance_calibration"):
        x_axis["Num of Sources"] = [1, 2, 3, 4, 5, 6]
        if RMSPE:
            Loss["SubNet+ESPRIT-MRA4"]  = np.array([0.0156, 0.0271 ,0.0518 ,0.0814, 0.0947, 0.1066])
            Loss["SubNet+R-MUSIC-MRA4"] = np.array([0.0169, 0.0356 ,0.0993 ,0.1444, 0.1174, 0.1066])
            Loss["ESPRIT-MRA4"]         = np.array([0.0349, 0.6459 ,0.5299 ,0.4614, 0.3685, 0.3091])
            Loss["MUSIC-MRA4"]          = np.array([0.0498, 0.3030 ,0.4766 ,1.3377, 1.5067, 1.5303])
            Loss["R-MUSIC-MRA4"]        = np.array([0.0351, 0.1814 ,0.5454 ,0.7044, 0.7115, 0.7196])
            Loss["SPS+R-MUSIC-MRA4"]    = np.array([0.0326, 0.4187 ,0.6080 ,0.7044, 0.7115, 0.6362])
            Loss["SPS+ESPRIT-MRA4"]     = np.array([0.2017, 0.5865 ,0.4798 ,0.3711, 0.3782, 0.3661])
            # Loss["SPS+MUSIC-MRA4"]      = np.array([0.0657, 0.5699 ,1.4091 ,0.6226, 0.5612, 0.5079])
            # Loss["CNN-MAR4"]            = np.array([])
            
            Loss["SubNet+ESPRIT-ULA7"]  = np.array([0.0099, 0.0152, 0.0220, 0.0283, 0.0336, 0.0510])
            Loss["SubNet+R-MUSIC-ULA7"] = np.array([0.0110, 0.0177, 0.0253, 0.0334, 0.0449, 0.0510])
            Loss["ESPRIT-ULA7"]         = np.array([0.0315, 0.0502, 0.0764, 0.1021, 0.1343, 0.1773])
            Loss["MUSIC-ULA7"]          = np.array([0.0444, 0.3701, 0.4361, 0.4343, 0.4143, 0.4292])
            Loss["R-MUSIC-ULA7"]        = np.array([0.0239, 0.0447, 0.0717, 0.1063, 0.1402, 0.1773])
            # Loss["SPS+R-MUSIC-ULA7"]    = np.array([0.0252, 0.0533, 0.1171, 0.7044, 0.7115, 0.6362])
            # Loss["SPS+ESPRIT-ULA7"]     = np.array([0.0251, 0.0495, 0.1171, 0.3740, 0.3746, 0.3654])
            # Loss["SPS+MUSIC-ULA7"]      = np.array([0.0476, 0.2931, 0.4621, 0.6226, 0.5612, 0.5079])
            # Loss["CNN-MAR4"]            = np.array([])
    
    elif simulation.startswith("sv_noise"):
        x_axis["sigma"] = [0.075, 0.1, 0.3, 0.4, 0.5, 0.75]
        if RMSPE:
            Loss["SubNet+R-MUSIC"]  = np.array([0.0148, 0.0173, 0.0304, 0.0408, 0.0506, 0.0807])
            Loss["SubNet+MUSIC"]    = np.array([0.0447, 0.0476, 0.0688, 0.0863, 0.0905, 0.1051])
            Loss["SubNet+ESPRIT"]   = np.array([0.0236, 0.0246, 0.0409, 0.0525, 0.0626, 0.0930])
            Loss["MUSIC"]           = np.array([0.0687, 0.0750, 0.1042, 0.1159, 0.1259, 0.1840])
            Loss["R-MUSIC"]         = np.array([0.0353, 0.0485, 0.0750, 0.0984, 0.1221, 0.1803])
            Loss["ESPRIT"]          = np.array([0.0370, 0.0455, 0.0860, 0.1112, 0.1323, 0.1905])
            Loss["CNN"]             = np.array([0.0391, 0.0472, 0.0631, 0.0929, 0.1166, 0.1329])

    elif simulation.startswith("sparse"):
        x_axis["Num of Sources"] = [1, 2, 3, 4, 5, 6]
        if mode.startswith("non-coherent"):            
            if RMSPE:
                Loss["SubNet+ESPRIT"]   = np.array([0.006635, 0.026476, 0.04648])
                Loss["SPS+R-MUSIC"]     = np.array([0.207308, 0.35327, 0.413619])
                Loss["SPS+MUSIC"]       = np.array([0.225392, 0.377007, 0.44180])
                Loss["SPS+ESPRIT"]      = np.array([0.31068, 0.35202, 0.413619])
                Loss["MUSIC"]           = np.array([0.01188, 0.05750, 0.847069])
                Loss["R-MUSIC"]         = np.array([0.09099, 0.58167, 0.91558])
                Loss["ESPRIT"]          = np.array([0.56074, 0.52576, 0.48113])
        elif mode.startswith("coherent"):            
            if RMSPE:
                Loss["SubNet+ESPRIT-MRA4"]  = np.array([0.0005, 0.0033, 0.0512, 0.1001, 0.1182, 0.1140])
                Loss["SubNet+ESPRIT-ULA7"]  = np.array([0.0006, 0.0016, 0.0145, 0.0181, 0.0335, 0.0677])
                Loss["SubNet+R-MUSIC-MRA4"] = np.array([0.0024, 0.1157, 0.1969, 0.1708, 0.1582, 0.1140])
                Loss["SubNet+R-MUSIC-ULA7"] = np.array([0.0009, 0.0105, 0.0329, 0.0719, 0.0857, 0.0677])
                Loss["ESPRIT-MRA4"]         = np.array([0.0025, 0.6141, 0.5283, 0.4610, 0.3685, 0.2844])
                # Loss["SPS+ESPRIT-MRA4"]     = np.array([0.0019, 0.5610, 0.3492, 0.3783, 0.3752, 0.3634])
                Loss["R-MUSIC-MRA4"]        = np.array([0.0003, 0.5678, 0.5687, 0.7049, 0.7119, 0.7170])
                # Loss["SPS+R-MUSIC-MRA4"]    = np.array([0.0004, 0.5565, 0.3613, 0.7049, 0.7119, 0.6356])
                Loss["ESPRIT-ULA7"]         = np.array([0.0004, 0.4471, 0.4091, 0.3658, 0.3240, 0.2833])
                Loss["R-MUSIC-ULA7"]        = np.array([0.0003, 0.2137, 0.3848, 0.3591, 0.3110, 0.2833])
                # Loss["SPS+R-MUSIC-ULA7"]    = np.array([0.0003, 0.0009, 0.0156, 0.7049, 0.7119, 0.6356])
                Loss["MUSIC-ULA7"]          = np.array([0.0022, 0.1701, 0.3684, 0.3656, 0.3626, 0.3725])
                # Loss["SPS+ESPRIT-ULA7"]     = np.array([0.0003, 0.0009, 0.0156, 0.3754, 0.3753, 0.3603])
                # Loss["MUSIC-MRA4"]          = np.array([0.0022, 0.4397, 0.4744, 0.7440, 1.5531, 1.6061])
        if mode.startswith("spacing_deviation"):            
            if RMSPE:
                Loss["SubNet+ESPRIT-MRA4"]  = np.array([0.0156, 0.0271 ,0.0518 ,0.0814, 0.0947, 0.1066])
                Loss["SubNet+ESPRIT-ULA7"]  = np.array([0.0099, 0.0152, 0.0220, 0.0283, 0.0336, 0.0510])
                Loss["SubNet+R-MUSIC-MRA4"] = np.array([0.0169, 0.0356 ,0.0993 ,0.1444, 0.1174, 0.1066])
                Loss["SubNet+R-MUSIC-ULA7"] = np.array([0.0110, 0.0177, 0.0253, 0.0334, 0.0449, 0.0510])
                Loss["ESPRIT-MRA4"]         = np.array([0.0349, 0.6459 ,0.5299 ,0.4614, 0.3685, 0.3091])
                Loss["R-MUSIC-MRA4"]        = np.array([0.0351, 0.1814 ,0.5454 ,0.7044, 0.7115, 0.7196])
                # Loss["MUSIC-MRA4"]          = np.array([0.0498, 0.3030 ,0.4766 ,1.3377, 1.5067, 1.5303])
                # Loss["SPS+R-MUSIC-MRA4"]    = np.array([0.0326, 0.4187 ,0.6080 ,0.7044, 0.7115, 0.6362])
                # Loss["SPS+ESPRIT-MRA4"]     = np.array([0.2017, 0.5865 ,0.4798 ,0.3711, 0.3782, 0.3661])
                Loss["ESPRIT-ULA7"]         = np.array([0.0315, 0.0502, 0.0764, 0.1021, 0.1343, 0.1773])
                # Loss["MUSIC-ULA7"]          = np.array([0.0444, 0.3701, 0.4361, 0.4343, 0.4143, 0.4292])
                Loss["R-MUSIC-ULA7"]        = np.array([0.0239, 0.0447, 0.0717, 0.1063, 0.1402, 0.1773])

        if mode.startswith("zoom_spacing_deviation"):            
            if RMSPE:
                Loss["SubNet+ESPRIT-MRA4"]  = np.array([0.0156, 0.0271 ,0.0518 ,0.0814, 0.0947, 0.1066])
                Loss["SubNet+ESPRIT-ULA7"]  = np.array([0.0099, 0.0152, 0.0220, 0.0283, 0.0336, 0.0510])
                Loss["SubNet+R-MUSIC-ULA7"] = np.array([0.0110, 0.0177, 0.0253, 0.0334, 0.0449, 0.0510])
                Loss["SubNet+R-MUSIC-MRA4"] = np.array([0.0169, 0.0356 ,0.0993 ,0.1444, 0.1174, 0.1066])
                Loss["ESPRIT-ULA7"]         = np.array([0.0315, 0.0502, 0.0764, 0.1021, 0.1343, 0.1773])
                Loss["R-MUSIC-ULA7"]        = np.array([0.0239, 0.0447, 0.0717, 0.1063, 0.1402, 0.1773])
                
                # Loss["SubNet+R-MUSIC-ULA7"] = np.array([0.0110, 0.0177, 0.0253, 0.0334, 0.0449, 0.0510])
    return x_axis, Loss

def plot(x_axis, Loss, conv_method, algorithm="all"):
    notations = {"MUSIC-ULA7"           : {"linestyle":'dashed',  "marker":'D', "color":'#104E8B'},
                 "SPS+MUSIC-MRA4"       : {"linestyle":'dashdot',  "marker":'x', "color":'#0f83f5'},
                 "SubNet+R-MUSIC-ULA7"  : {"linestyle":'dashed', "marker":'>', "color":'#1a05e3'},
                 "R-MUSIC-MRA4"         : {"linestyle":'solid',  "marker":'s', "color":'#006400'},
                 "R-MUSIC-ULA7"         : {"linestyle":'solid',  "marker":'*', "color":'#456216'},
                 "SPS+R-MUSIC-MRA4"     : {"linestyle":'dashdot',  "marker":'*', "color":'#0d8074'},
                 "SubNet+R-MUSIC-MRA4"  : {"linestyle":'dashdot', "marker":'>', "color":'#039403'},
                 "ESPRIT-MRA4"          : {"linestyle":'solid',  "marker":'o', "color":'#842ab0'},
                 "ESPRIT-ULA7"          : {"linestyle":'solid',  "marker":'p', "color":'#754130'},
                 "SPS+ESPRIT-MRA4"      : {"linestyle":'dashdot',  "marker":'*', "color":'#9f59c2'},
                 "SubNet+ESPRIT-MRA4"   : {"linestyle":'dashdot', "marker":'>', "color":'#BF3EFF'},
                 "MUSIC-MRA4"           : {"linestyle":'solid', "marker":'x', "color":'#FFA500'},
                 "SubNet+ESPRIT-ULA7"   : {"linestyle":'dashed', "marker":'>', "color":'#D49137'},
                 "CNN"                  : {"linestyle":'dashdot', "marker":'h', "color":'#E408C3'}
                }
    
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    plt.style.use('plotting\plot_style.txt')
    for axis_name, axis_array in x_axis.items():
        axis_name = axis_name
        axis_array = axis_array
        
    for method, loss in Loss.items():
        if algorithm == "all" or algorithm in method:
            if algorithm == "MUSIC" and "R-MUSIC" in method:
                pass
            else:
                plt.plot(axis_array, conv_method(loss), linestyle = notations[method]["linestyle"],
                        marker=notations[method]["marker"], label=method, color=notations[method]["color"])
        elif algorithm == "coherent":
            if "SPS" in method or "SubNet" in method:
                plt.plot(axis_array, conv_method(loss), linestyle = notations[method]["linestyle"],
                        marker=notations[method]["marker"], label=method, color=notations[method]["color"])
        else:
            pass
            
    
    plt.xlim([np.min(axis_array) - 0.1 * np.abs(np.min(axis_array)), np.max(axis_array) + 0.1 * np.abs(np.min(axis_array))])
    # plt.ylim([-45, -15])
    plt.xlabel(axis_name)
    if conv_method == unit:
        plt.ylabel("RMSE [rad]")
    elif conv_method == rad2dB:
        plt.ylabel("MSE [dB]")
    plt.legend()

if __name__ == "__main__":
    # scenario_to_plot(simulation="distance_calibration")
    # T = 20
    # mode = "non-coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode=mode, T = T)

    ## ESPRIT
    # algorithm="ESPRIT"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.ylim([-24, -5])
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    # plt.xlim([4.9, 10.1])
    
    ## MUSIC
    # algorithm="MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-20, -9])
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    ## R-MUSIC
    # algorithm="R-MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -6])
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')

    # ## Coherent
    # algorithm="coherent"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -10])
    # # plt.legend(bbox_to_anchor=(0.41, 0.34), loc=0)
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')

    ## All
    # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.xscale("log", base=10)
    # plt.ylim([0.015, 0.63])
    # plt.ylim([0.015, 0.23])
    # plt.xlim([47, 1050])
    # plt.legend(fontsize='x-small', bbox_to_anchor=(0.21, 0.57))
    # plt.legend(fontsize='small', bbox_to_anchor=(0.21, 0.5))
    # plt.legend(bbox_to_anchor=(0.41, 0.34))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    # plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    # plt.savefig("{}_{}_{}_no_classical_methods.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode=mode, T = T)
    # ## ESPRIT
    # algorithm="ESPRIT"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-24, -4])
    # plt.legend(bbox_to_anchor=(0.5, 0.5))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    ## distance_calibration
    # plt.xlim([4.9, 10.1])
    
    #######################################################
    # simulation = "SNR"
    # conv_method = rad2dB
    # mode = "non-coherent"
    # T = 200
    # algorithm="all"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=conv_method, mode=mode, T=T)
    # plot(x_axis, Loss, conv_method=conv_method, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-33.5, -10.5])
    # plt.legend()
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################
    
    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 200
    # mode = "coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([-5.1, -0.9])
    # plt.ylim([-36, -5])
    # # plt.legend(bbox_to_anchor=(0.21, 0.34), loc=0)
    # plt.legend()
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################

    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 20
    # mode = "coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-41, -5])
    # # plt.legend(bbox_to_anchor=(0.41, 0.54), loc=0)
    # plt.legend()
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################
    
    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 2
    # mode = "coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-26, -4.5])
    # # plt.legend(bbox_to_anchor=(0.41, 0.54), loc=0)
    # plt.legend()
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################

    ########################################################
    # simulation = "SNR"
    # algorithm="all"
    # T = 2
    # mode = "non-coherent"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=rad2dB, mode= mode, T=T)
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # # plt.xlim([-5.1, -0.9])
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-25.5, -4.5])
    # # plt.legend(bbox_to_anchor=(0.41, 0.54), loc=0)
    # plt.legend()
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    ########################################################

    # ########################################################
    # simulation = "distance_calibration"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.ylim([0.015, 0.27])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    # ########################################################

    # ########################################################
    mode = "spacing_deviation"
    simulation = "sparse"
    x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    algorithm="all"
    plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # plt.xlabel(r"$\eta [\lambda / 2]$")
    plt.xlim([0.9, 6.1])
    # plt.xscale("log", base=10)
    # plt.ylim([0, 1.5])
    plt.ylim([0, 0.72])
    plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')

    # ########################################################
    # mode = "zoom_spacing_deviation"
    # simulation = "sparse"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.xlim([0.9, 6.1])
    # # plt.xscale("log", base=10)
    # # plt.ylim([0, 1.5])
    # plt.ylim([0.005, 0.18])
    # plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')

    ########################################################
    mode = "coherent"
    simulation = "sparse"
    x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    algorithm="all"
    plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # plt.xlabel(r"$\eta [\lambda / 2]$")
    plt.xlim([0.9, 6.1])
    # plt.xscale("log", base=10)
    # plt.ylim([0, 1.5])
    plt.ylim([0.00005, 0.72])
    plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')

    # mode = "coherent"
    # simulation = "sparse"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.xlim([1.9, 4.1])
    # # plt.xscale("log", base=10)
    # # plt.ylim([0.015, 0.62])
    # # plt.ylim([0.02, 0.29])
    # plt.ylim([0, 0.95])
    # plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    ########################################################

    ########################################################
    # mode = "non-coherent"
    # simulation = "OFDM"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.xlim([47, 1050])
    # plt.xscale("log", base=10)
    # plt.ylim([0.015, 0.62])
    # # plt.ylim([0.02, 0.29])
    # plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    ########################################################

    # ########################################################
    # mode = "coherent"
    # simulation = "OFDM"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit, mode=mode)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # # plt.xlabel(r"$\eta [\lambda / 2]$")
    # plt.xlim([47, 1050])
    # plt.xscale("log", base=10)
    # plt.ylim([0.015, 0.67])
    # # plt.ylim([0.02, 0.29])
    # plt.savefig("{}_{}_{}.pdf".format(simulation, mode, algorithm),bbox_inches='tight')
    # ########################################################

    ########################################################
    # simulation = "sv_noise"
    # x_axis, Loss = scenario_to_plot(simulation=simulation, conv_method=unit)
    # algorithm="all"
    # plot(x_axis, Loss, conv_method=unit, algorithm=algorithm)
    # plt.xlabel(r"$\sigma^2_{\rm sv}$")
    # plt.xlim([0.065, 0.76])
    # # plt.xscale("log", base=10)
    # # plt.ylim([0.02, 0.68])
    # plt.ylim([0.01, 0.195])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    ########################################################
    
    
    
    
    
    # algorithm="MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-16, -5])
    # plt.legend(bbox_to_anchor=(0.5, 0.5))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    # ## R-MUSIC
    # algorithm="R-MUSIC"
    # plot(x_axis, Loss, conv_method=rad2dB, algorithm=algorithm)
    # plt.xlim([4.9, 10.1])
    # plt.ylim([-26, -4])
    # plt.legend(bbox_to_anchor=(0.5, 0.5))
    # plt.savefig("{}_T={}_{}_{}.pdf".format(simulation, T, mode, algorithm),bbox_inches='tight')
    
    # plt.yticks(np.arange(-35, -14.5, 2.5))
    # scenario_to_plot(simulation="SNR", conv_method=rad2dB, mode="non-coherent", T = 2)
    # scenario_to_plot(simulation="SNR", conv_method=rad2dB, mode="coherent", T = 2)
    # plt.show()

    #######################################################
    # simulation = "eigenvalues"
    empirical_cov = [[   1.32462396  +0.j        ,    1.37029507 +12.71751673j,
           3.19058897 +11.36444576j,   -3.62373953  +0.217819j  ,
         -14.07081456  -1.83435012j,   -9.18583814  +1.72975747j,
           0.35775326  -3.06552153j,    1.92934983 -12.16284328j],
       [   1.37029507 -12.71751673j,  123.51651923  +0.j        ,
         112.40894199 -18.87609266j,   -1.65743318 +35.01631097j,
         -32.1672774 +133.19419286j,    7.10459059 +89.98125679j,
         -29.06152625  -6.60595185j, -114.77777044 -31.10560002j],
       [   3.19058897 -11.36444576j,  112.40894199 +18.87609266j,
         105.18493554  +0.j        ,   -6.85966093 +31.61407584j,
         -49.62960083+116.30044207j,   -7.28546299 +82.97517489j,
         -25.43854764 -10.45314528j,  -99.70241736 -45.84895571j],
       [  -3.62373953  -0.217819j  ,   -1.65743318 -35.01631097j,
          -6.85966093 -31.61407584j,    9.94918839  +0.j        ,
          38.19152628  +7.33196595j,   25.41389852  -3.22154864j,
          -1.48278568  +8.32743969j,   -7.27811052 +32.95631695j],
       [ -14.07081456  +1.83435012j,  -32.1672774 -133.19419286j,
         -49.62960083-116.30044207j,   38.19152628  -7.33196595j,
         152.00741458  +0.j        ,   95.18115963 -31.09496836j,
           0.44492634 +33.05891425j,   -3.65130843+131.87155093j],
       [  -9.18583814  -1.72975747j,    7.10459059 -89.98125679j,
          -7.28546299 -82.97517489j,   25.41389852  +3.22154864j,
          95.18115963 +31.09496836j,   65.95961278  +0.j        ,
          -6.48400797 +20.79122767j,  -29.26224018 +81.82594155j],
       [   0.35775326  +3.06552153j,  -29.06152625  +6.60595185j,
         -25.43854764 +10.45314528j,   -1.48278568  -8.32743969j,
           0.44492634 -33.05891425j,   -6.48400797 -20.79122767j,
           7.19102929  +0.j        ,   28.66903396  +1.18008334j],
       [   1.92934983 +12.16284328j, -114.77777044 +31.10560002j,
         -99.70241736 +45.84895571j,   -7.27811052 -32.95631695j,
          -3.65130843-131.87155093j,  -29.26224018 -81.82594155j,
          28.66903396  -1.18008334j,  114.49071775  +0.j        ]]
    ssn_cov = [[  56731.0547+0.0000j,   -4314.5723-98431.1094j,
         -123549.7266+117745.7031j,   25086.9844+128895.8984j,
           64204.0430+39669.5938j, -111450.4844-118976.4844j,
         -100929.3750+28066.1016j,   30292.9414+13613.3086j],
        [  -4314.5723+98431.1094j,  833051.8750+0.0000j,
           29494.4316-299663.6250j, -180515.4688+362784.9375j,
          -60003.3320+566930.3125j,  316108.0312-2534.6602j,
         -278758.6250-286772.8438j,  -99148.2031+141427.1719j],
        [-123549.7266-117745.7031j,   29494.4316+299663.6250j,
          892673.0625+0.0000j,  123809.5547-337977.7188j,
         -187054.2188+115169.7891j,   56510.7383+824029.7500j,
          347948.0000-10480.0703j, -194997.6719-171329.5938j],
        [  25086.9844-128895.8984j, -180515.4688-362784.9375j,
          123809.5547+337977.7188j,  857850.5625+0.0000j,
          386227.1875-195151.2500j, -301688.6562+117594.9141j,
           -5905.2197+697171.1875j,  175821.2812+23194.7324j],
        [  64204.0430-39669.5938j,  -60003.3320-566930.3125j,
         -187054.2188-115169.7891j,  386227.1875+195151.2500j,
          683439.9375+0.0000j,   90717.5000-249569.8125j,
         -225344.0312+242898.7188j,   54673.5938+239450.0469j],
        [-111450.4844+118976.4844j,  316108.0312+2534.6602j,
           56510.7383-824029.7500j, -301688.6562-117594.9141j,
           90717.5000+249569.8125j,  944404.6875+0.0000j,
           44709.7578-305063.6250j, -139964.2344+200239.0469j],
        [-100929.3750-28066.1016j, -278758.6250+286772.8438j,
          347948.0000+10480.0703j,   -5905.2197-697171.1875j,
         -225344.0312-242898.7188j,   44709.7578+305063.6250j,
          710608.1250+0.0000j,  -40602.5508-164594.3125j],
        [  30292.9414-13613.3086j,  -99148.2031-141427.1719j,
         -194997.6719+171329.5938j,  175821.2812-23194.7324j,
           54673.5938-239450.0469j, -139964.2344-200239.0469j,
          -40602.5508+164594.3125j,  226326.3438+0.0000j]]
    sps_cov = [[ 59.99381678 +0.j        ,  36.2777756  +8.19686646j,
         -5.67063663+39.86491254j, -11.1398164 +56.17865661j,
         -9.92072053+27.66251958j],
       [ 36.2777756  -8.19686646j,  97.66451444 +0.j        ,
         59.73049174 -2.75625481j,  -6.35705229+45.28852966j,
        -11.14670862+89.09208959j],
       [ -5.67063663-39.86491254j,  59.73049174 +2.75625481j,
         83.27528782 +0.j        ,  30.00725425 +7.16057528j,
        -13.25825404+56.99093731j],
       [-11.1398164 -56.17865661j,  -6.35705229-45.28852966j,
         30.00725425 -7.16057528j,  58.77681126 +0.j        ,
         38.88942797 -0.44792285j],
       [ -9.92072053-27.66251958j, -11.14670862-89.09208959j,
        -13.25825404-56.99093731j,  38.88942797 +0.44792285j,
         84.9121936  +0.j        ]]
    # Deep_RootMUSIC_Rx = [[[162162.7500+0.0000j, -53419.5820-29567.0664j,
    # 72869.0625+45681.9219j,  18064.2520-29616.1172j,
    # 138235.3750-11426.2324j, -69004.9141-33662.0234j,
    # 101304.8906+35886.5547j, -31843.0645-50518.2773j],
    # [-53419.5820+29567.0664j, 130456.1172+0.0000j,
    # -38763.6680-52971.1641j,  78656.8203+35861.0977j,
    # -21711.6445-38134.4062j,  56236.5742-67059.6250j,
    # -74090.4531+10358.3154j,  61832.4805+10086.8291j],
    # [ 72869.0625-45681.9219j, -38763.6680+52971.1641j,
    # 172605.5938+0.0000j,  -3078.1484-31445.6250j,
    # 114771.0078-7304.6855j,  57420.8672-17480.7793j,
    # 67166.2031-75923.6250j, -69415.0156+20219.1660j],
    # [ 18064.2520+29616.1172j,  78656.8203-35861.0977j,
    # -3078.1484+31445.6250j, 123215.0781+0.0000j,
    # 8936.6953-19623.5977j,  31236.8535-37875.9219j,
    # 20523.3242+31001.5547j,  24692.6172-31873.0938j],
    # [138235.3750+11426.2324j, -21711.6445+38134.4062j,
    # 114771.0078+7304.6855j,   8936.6953+19623.5977j,
    # 193280.2812+0.0000j,   -224.4319-61926.8867j,
    # 60970.2070+7851.4541j, -16777.9082-12268.1699j],
    # [-69004.9141+33662.0234j,  56236.5742+67059.6250j,
    # 57420.8672+17480.7793j,  31236.8535+37875.9219j,
    # -224.4319+61926.8867j, 150367.0000+0.0000j,
    # -41780.7305-50633.0234j,  10084.9014+38179.3789j],
    # [101304.8906-35886.5547j, -74090.4531-10358.3154j,
    # 67166.2031+75923.6250j,  20523.3242-31001.5547j,
    # 60970.2070-7851.4541j, -41780.7305+50633.0234j,
    # 134363.7500+0.0000j, -69894.4453-33309.3672j],
    # [-31843.0645+50518.2773j,  61832.4805-10086.8291j,
    # -69415.0156-20219.1660j,  24692.6172+31873.0938j,
    # -16777.9082+12268.1699j,  10084.9014-38179.3789j,
    # -69894.4453+33309.3672j,  78712.8281+0.0000j]]]

    # M = 3
    # empirical_eig = np.sort(np.real(LA.eigvals(empirical_cov)))[::-1]
    # norm_empirical_eig = empirical_eig / np.max(empirical_eig)
    
    # sps_eig = np.sort(np.real(LA.eigvals(sps_cov)))[::-1]
    # norm_sps_eig = sps_eig / np.max(sps_eig)
    
    # ssn_eig = np.sort(np.real(LA.eigvals(ssn_cov)))[::-1]
    # norm_ssn_eig = ssn_eig / np.max(ssn_eig)
    
    # algorithm = "ssn"
    # plt.style.use('default')
    # fig = plt.figure(figsize=(7, 5.5))
    # plt.style.use('plot_style.txt')

    # plt.xlabel("Index")
    # plt.ylabel("Eigenvalues [λ]")
    # plt.xlim([0.85, 8.15])
    # plt.ylim([-0.02, 1.02])
    
    # markerline, stemlines, baseline = plt.stem([i + 1 + 0.05 for i in range(empirical_eig.shape[0])],norm_empirical_eig, '#842ab0', label="Empirical")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # plt.setp(stemlines, 'linestyle', 'dashed')
    # markerline, stemlines, baseline = plt.stem([i + 1 + 0.05 for i in range(norm_sps_eig.shape[0])],norm_sps_eig, '#0f83f5', label="SPS")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # plt.setp(stemlines, 'linestyle', 'dashed')
    # markerline, stemlines, baseline = plt.stem([i + 1 - 0.05 for i in range(norm_ssn_eig.shape[0])], norm_ssn_eig,'#039403', markerfmt='>', label="SubNet")
    # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # plt.legend()
    # plt.savefig("eigenvalues.pdf",bbox_inches='tight')

    # # markerline, stemlines, baseline = plt.stem(x, y, markerfmt='o', label='pcd')
    # # plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    # # plt.setp(stemlines, 'linestyle', 'dotted')

    # algorithm = "rm"
    # plt.style.use('default')
    # fig = plt.figure(figsize=(7, 5.5))
    # plt.style.use('plot_style.txt')

    # plt.xlabel("Index")
    # plt.ylabel("Eigenvalues [λ]")
    # plt.xlim([0.9, 8.1])
    # plt.ylim([-0.02, 1.02])
    # plt.ylim([-10000, 6 * 100000])
    # plt.savefig("{}_{}.pdf".format(simulation, algorithm),bbox_inches='tight')
    
    
    plt.show()
    print("hi!")