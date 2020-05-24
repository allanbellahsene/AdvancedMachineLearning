%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=======================================================================================================%
%====================================== Advanced Machine Learning ======================================%
%========================================= Team O - SVR vs LWPR ========================================%
%==================================== BRODARD Lionel, BELLAHSENE Allan =================================%w
%============================================== Read Me ================================================%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We did this project for the Advanced Machine Learning course at EPFL. The goal of the project is to
Compare the SVR and LWPR algorithm using financial data. 
Please see the report for further information concerning the results and the methodology.

######################################## 0. Setup #######################################

The code for the LWPR is done in Matlab, and for the SVR in python.
Our code uses different external libraries (numpy, scipy, pandas, scikit-learn, ...) and the LWPR code
Made available by Prof. Billard (EPFL) based on the work of Schaal et al. (2000)

####################################### I. Structure ######################################
#######################################################################################
The parent folder called "AdvancedMachineLearning" contains three folders and 1 file:

1) code: a folder containing the LWPR code in Matlab

2) SVR: a folder containing the SVR code in Python (jupyter notebook), the data used, as well as the figures   

3) data: a folder containing the data (basic data and results)

4) figures: a folder containing all the figures

5) README.md: a folder containing the report

6) report.pdf: the report
#######################################################################################
The code folder is divided into 1 folder and 5 files:

1) CPU_dimension.m: a Matlab file used to create the figure on CPU consumption when dimension increases

2) dataPreprocessing.m: Matlab file for the pre-processing and cleaning of data

3) dataVisualisation.m: Matlab file for basic statistics and visualisation of the data

4) functions: folder containing all the LWPR functions (lwpr.m given by Prof. Billard EPFL)

5) main.m: main Matlab file for LWPR predictions

#######################################################################################
The 'SVR' folder contains 15 files.

1) SupportVectorRegression_Implementation.ipynb: a Jupyter notebook file containing the code in which we implemented the SVR algorithm 

2) financial_data.csv: a file containing the preprocessed data. The preprocessing was done in Matlab, which was exported from 'dataPreprocessing.m'

3) predictions_LWPR_opti.csv: a file containing the best predictions of the LWPR algorithm, which was exported from 'main.m'

4) 12 '.pdf' files, which are the figures of the SVR implementation.
