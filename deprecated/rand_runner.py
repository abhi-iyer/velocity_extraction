from imports import *

from models import *
from data import *
from utils import *

from rand_configs import RAND_CONFIGS
from rand_experiment import *


# shift1_rand_LR1_mse= PINV_Experiment("shift1_rand_LR1_mse", RAND_CONFIGS["shift1_rand_LR1_mse"])
# shift1_rand_LR1_mse.train(num_epochs=250, show_plot=False)

# shift1_rand_LR1_croppedmse= PINV_Experiment("shift1_rand_LR1_croppedmse", RAND_CONFIGS["shift1_rand_LR1_croppedmse"])
# shift1_rand_LR1_croppedmse.train(num_epochs=250, show_plot=False)



# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_1", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=500, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_2", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=500, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_3", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=500, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_4", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_5", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_6", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_7", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_8", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_9", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_10", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_11", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_12", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_Base_13", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_Base"]).train(num_epochs=200, show_plot=False)

# Rand_Experiment("shift1_rand_LR1_Noise1_MSE_ContrastV_Base_1", RAND_CONFIGS["shift1_rand_LR1_Noise1_MSE_ContrastV_Base"]).train(num_epochs=500, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_Noise1_MSE_ContrastV_Base_2", RAND_CONFIGS["shift1_rand_LR1_Noise1_MSE_ContrastV_Base"]).train(num_epochs=500, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_Noise1_MSE_ContrastV_Base_3", RAND_CONFIGS["shift1_rand_LR1_Noise1_MSE_ContrastV_Base"]).train(num_epochs=500, show_plot=False)

# Rand_Experiment("shift1_rand_LR1_Noise2_MSE_ContrastV_Base_1", RAND_CONFIGS["shift1_rand_LR1_Noise2_MSE_ContrastV_Base"]).train(num_epochs=300, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_Noise2_MSE_ContrastV_Base_2", RAND_CONFIGS["shift1_rand_LR1_Noise2_MSE_ContrastV_Base"]).train(num_epochs=300, show_plot=False)
# Rand_Experiment("shift1_rand_LR1_Noise2_MSE_ContrastV_Base_3", RAND_CONFIGS["shift1_rand_LR1_Noise2_MSE_ContrastV_Base"]).train(num_epochs=300, show_plot=False)


#Rand_Experiment("shift1_rand_LR1_MSE_ContrastV1_1", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV1"]).train(num_epochs=500, show_plot=False)

#Rand_Experiment("shift1_rand_LR1_Noise1_MSE_ContrastV1_1", RAND_CONFIGS["shift1_rand_LR1_Noise1_MSE_ContrastV1"]).train(num_epochs=1000, show_plot=False)

Rand_Experiment("shift1_rand_LR1_NoiseSweep1_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep1"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep2_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep2"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep3_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep3"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep4_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep4"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep5_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep5"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep6_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep6"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep7_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep7"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep8_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep8"]).train(num_epochs=800, show_plot=False)
Rand_Experiment("shift1_rand_LR1_NoiseSweep9_1", RAND_CONFIGS["shift1_rand_LR1_NoiseSweep9"]).train(num_epochs=800, show_plot=False)



# shift1_rand_LR1_MSE_ContrastV_2 = Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_2", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_2"])
# shift1_rand_LR1_MSE_ContrastV_2.train(num_epochs=500, show_plot=False)

# shift1_rand_LR1_MSE_ContrastV_3 = Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_3", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_3"])
# shift1_rand_LR1_MSE_ContrastV_3.train(num_epochs=500, show_plot=False)

# shift1_rand_LR1_MSE_ContrastV_4 = Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_4", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_4"])
# shift1_rand_LR1_MSE_ContrastV_4.train(num_epochs=500, show_plot=False)

# shift1_rand_LR1_MSE_ContrastV_5 = Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_5", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_5"])
# shift1_rand_LR1_MSE_ContrastV_5.train(num_epochs=500, show_plot=False)

# shift1_rand_LR1_MSE_ContrastV_6 = Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_6", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_6"])
# shift1_rand_LR1_MSE_ContrastV_6.train(num_epochs=500, show_plot=False)

# shift1_rand_LR1_MSE_ContrastV_7 = Rand_Experiment("shift1_rand_LR1_MSE_ContrastV_7", RAND_CONFIGS["shift1_rand_LR1_MSE_ContrastV_7"])
# shift1_rand_LR1_MSE_ContrastV_7.train(num_epochs=500, show_plot=False)

# shift1_rand_LR2_mse = Rand_Experiment("shift1_rand_LR2_mse", RAND_CONFIGS["shift1_rand_LR2_mse"])
# shift1_rand_LR2_mse.train(num_epochs=500, show_plot=False)