from enum import Enum
from shutil import rmtree
import os


### CONSTANTS, HYPERPARAMETERS AND SHARED FUNCTIONS


# Enumeration to select transformation of raw data.
class DataTransform(Enum):
    PERCENTAGE_RETURN = 1
    LOG_RETURN = 2
    NONE = 3


# Enumeration to select neural network activations.
#
class ActivationType(Enum):
    TANH = 10
    RELU = 20
    ELU = 30
    LEAKY_RELU = 40


# Enumeration for neural network optimizer
#
class OptimizerType(Enum):
    ADAM = 100
    RMSPROP = 200


X_SAVE_PATH = '../data/data_saves/'
RESULTS_USED_DIR = '../results_saves/ln_and_standardized_VIX/final_architecture_100ep_80mn_32btch_256_256_4times/'

# data info
USE_SAVED_X = True# if False, won't load data from 'weekly_returns' .py file

# data info
TRADING_DAYS_YEAR = 252
DATA_TRANSFORM = DataTransform.PERCENTAGE_RETURN  # using percentage returns because chaining returns for charts
WEEKLY = True  # can change daily data to weekly
SAMPLE_LEN = 52  # here, should be: (a) =52 for weekly data or, (b) =TRADING_DAYS_YEAR for daily
N_SERIES = 3
N_CONDITIONALS = 2
N_INPUTS = N_SERIES + N_CONDITIONALS

# settings for random generations
RAND_MEAN = 0.
RAND_STD = 1.

# optimization settings
EPOCHS = 100
MIN_EPOCHS = 80
L2_REGULARIZATION = 0.001
DROPOUT = 0.5
KEEP_PROB = 1.0 - DROPOUT
BATCH_SIZE = 32
GENERATOR_SCOPE = "GAN/Generator"
DISCRIMINATOR_SCOPE = "GAN/Discriminator"
ACTIVATION = ActivationType.LEAKY_RELU
GENERATOR_OPTIMIZER = OptimizerType.ADAM
DISCRIMINATOR_OPTIMIZER = OptimizerType.ADAM
GENERATOR_TRAINS_PER_BATCH = 1
DISCRIMINATOR_TRAINS_PER_BATCH = 1
DISCRIMINATOR_RANDOM_SCHEDULE = 0.25
GENERATOR_DROPOUT_ALWAYS_ON = True
LAYER_NORM = False
DISCRIMINATOR_RNN = True
DISCRIMINATOR_CONVOLUTION = False
RNN_NEURONS = [256, 256]


def delete_files_in_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)  # remove dir and all contains
        except Exception as exc:
            print(exc)
