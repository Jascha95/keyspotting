# global NEURONS_PER_CORE, CORES_PER_CHIP, NEURONS_PER_CHIP, MAX_NUM_CAMS, NUM_CHIPS
import numpy as np
import os

NEURONS_PER_CORE = 256
CORES_PER_CHIP = 4
NEURONS_PER_CHIP = NEURONS_PER_CORE * CORES_PER_CHIP
MAX_NUM_CAMS = 64
NUM_CHIPS = 4

ADDR_NUM_BITS = 15
ISI_NUM_BITS = 16
MAX_ISI = 2**ISI_NUM_BITS-1
MAX_FPGA_LEN = 2**ADDR_NUM_BITS-1

map_file_path = os.path.join(os.path.dirname(__file__) + "/details/linear_fine_coarse_bias_map.npy")
LINEAR_BIAS_MAP = np.load(map_file_path).astype(int)

LINEAR_BIAS_MAX = int(24e+08)
LINEAR_BIAS_MIN = 0

BIAS_NAMES = ['IF_AHTAU_N', 'IF_AHTHR_N', 'IF_AHW_P', 'IF_BUF_P', 'IF_CASC_N', 'IF_DC_P', 'IF_NMDA_N', 'IF_RFR_N', 'IF_TAU1_N', 'IF_TAU2_N',
              'IF_THR_N', 'NPDPIE_TAU_F_P', 'NPDPIE_TAU_S_P', 'NPDPIE_THR_F_P', 'NPDPIE_THR_S_P', 'NPDPII_TAU_F_P', 'NPDPII_TAU_S_P', 'NPDPII_THR_F_P',
              'NPDPII_THR_S_P', 'PS_WEIGHT_EXC_F_N', 'PS_WEIGHT_EXC_S_N', 'PS_WEIGHT_INH_F_N', 'PS_WEIGHT_INH_S_N', 'PULSE_PWLK_P', 'R2R_P']

BIAS_NAMES_LOOKUP = dict([[name, i] for i, name in enumerate(BIAS_NAMES)])

BIAS_NAMES_FULL = ['Adaptation time constant', 'Adaptation gain', 'Adaptation weight', 'V_mem monitoring filter', 'Cascode (enable adaptation)', 'DC input level', 'NMDA mechanism enable', 'Refractory period (neuron)', 'Tau 1 (neuron time constant)', 'Tau 2 (neuron time constant)',
                       'Neuron membrane voltage gain', 'AMPA synapse time constant', 'NMDA synapse time constant', 'AMPA synapse gain', 'NMDA synapse gain',
                       'GABA A synapse time constant', 'GABA B synapse time constant', 'GABA A synapse gain', 'GABA B synapse gain',
                       'AMPA synapse weight', 'NMDA synapse weight', 'GABA A synapse weight', 'GABA B synapse weight', 'Spike pulse length', 'R2R']

BIAS_NAMES_SHORT = ['AHP tau', 'AHP gain', 'AHP wgt', 'V mon buf', 'Casc', 'DC', 'NMDA enable', 'Rfr', 'Tau 1', 'Tau 2',
                       'V_mem gain', 'AMPA tau', 'NMDA tau', 'AMPA gain', 'NMDA gain',
                       'GABA_A tau', 'GABA_B tau', 'GABA_A gain', 'GABA_B gain',
                       'AMPA wgt', 'NMDA wgt', 'GABA_A wgt', 'GABA_B wgt', 'PWLK', 'R2R']


BIAS_TAGS = ['ahp_tau', 'ahp_gain', 'ahp_w', 'buf', 'casc', 'dc', 'nmda', 'rfr', 'tau1', 'tau2',
                       'gain', 'ampa_tau', 'nmda_tau', 'ampa_gain', 'nmda_gain',
                       'gaba_a_tau', 'gaba_b_tau', 'gaba_a_gain', 'gaba_b_gain',
                       'ampa_w', 'nmda_w', 'gaba_a_w', 'gaba_b_w', 'pwlk', 'r2r']

BIAS_TAGS_LOOKUP = dict([[name, i] for i, name in enumerate(BIAS_TAGS)])