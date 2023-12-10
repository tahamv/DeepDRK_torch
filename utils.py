import os


DATA_FOLDER = 'data'
TEST_TCGA_DATA_FOLDER = os.path.join(DATA_FOLDER, 'TCGA_test_data')
RAW_BOTH_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CTRP_GDSC_data')
DRUG_DATA_FOLDER = os.path.join(DATA_FOLDER, 'drug_data')
GDSC_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'GDSC_data')
CCLE_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CCLE_data')
CTRP_RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'CTRP_data')
SIM_DATA_FOLDER = os.path.join(DATA_FOLDER, 'similarity_data')

GDSC_SCREENING_DATA_FOLDER = os.path.join(GDSC_RAW_DATA_FOLDER, 'drug_screening_matrix_GDSC.tsv')
CCLE_SCREENING_DATA_FOLDER = os.path.join(CCLE_RAW_DATA_FOLDER, 'drug_screening_matrix_ccle.tsv')
CTRP_SCREENING_DATA_FOLDER = os.path.join(CTRP_RAW_DATA_FOLDER, 'drug_screening_matrix_ctrp.tsv')
BOTH_SCREENING_DATA_FOLDER = os.path.join(RAW_BOTH_DATA_FOLDER, 'drug_screening_matrix_gdsc_ctrp.tsv')

CTRP_FOLDER = os.path.join(DATA_FOLDER, 'CTRP')
GDSC_FOLDER = os.path.join(DATA_FOLDER, 'GDSC')
CCLE_FOLDER = os.path.join(DATA_FOLDER, 'CCLE')

MODEL_FOLDER = os.path.join(DATA_FOLDER, 'model')

TCGA_DATA_FOLDER = os.path.join(DATA_FOLDER, 'TCGA_data')
TCGA_SCREENING_DATA = os.path.join(TCGA_DATA_FOLDER, 'TCGA_screening_matrix.tsv')

BUILD_SIM_MATRICES = True  # Make this variable True to build similarity matrices from raw data
SIM_KERNEL = {'cell_CN': ('euclidean', 0.001), 'cell_exp': ('euclidean', 0.01), 'cell_methy': ('euclidean', 0.1),
              'cell_mut': ('jaccard', 1), 'drug_DT': ('jaccard', 1), 'drug_comp': ('euclidean', 0.001),
              'drug_desc': ('euclidean', 0.001), 'drug_finger': ('euclidean', 0.001)}
SAVE_MODEL = False  # Change it to True to save the trained model
VARIATIONAL_AUTOENCODERS = False
# DATA_MODALITIES=['cell_CN','cell_exp','cell_methy','cell_mut','drug_comp','drug_DT'] # Change this list to only consider specific data modalities
DATA_MODALITIES = ['cell_exp', 'drug_desc']
RANDOM_SEED = 42  # Must be used wherever can be used



