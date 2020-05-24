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

1) code:  folder containing the code in Matlab and Python (notebook)

2) data: a folder containing the data (basic data and results)

3) figures: a folder containing all the figures

4) README.md: a folder containing the report

5) report.pdf: the report
#######################################################################################
The code folder is divided into 1 folder and 5 files:

1) CPU_dimension.m: a Matlab file used to create the figure on CPU consumption when dimension increases

2) dataPreprocessing.m: Matlab file for the pre-processing and cleaning of data

3) dataVisualisation.m: Matlab file for basic statistics and visualisation of the data

4) functions: folder containing all the LWPR functions (lwpr.m given by Prof. Billard EPFL)

5) main.m: main Matlab file for LWPR predictions

6) SVR.ipnyb: Jupyter notebook for the SVR predictions
