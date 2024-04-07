# NLB_P3_ExaminingTemporalChanges

This repository contains code necessary to ptoduce results for the paper "Examining Temporal Changes in " to be published
The Cardiovascular modelparameters must initially be estimated before analaysis can be performed. 

1. All folders in the root directory starting with "OpenLoop_..." or "ClosedLoop_..." must have the file "RealDataEstimatorTwoStep..."-.py to be run to produce estimates. 

2. The resulting "MultiEstimateRound2..."-.csv-files, files that are produced must be copied along with the "ZaoFits.csv"-file, to the dubdirectory "ParameterVisualization/".

3. Within this directory, the file "AvgComputerPercentiles.py" must be run in order to refomrat output files. 

4. If analysis upon the measurements after CPET also is to be performed, the steps 1.-3. must be repeated for the directories within the "postCPET"-directory in the root.

5. In the analysis folder, run all files "Annotate_..."-.py, to structure the 

6. Now analysis can be performed within the "Analysis folder" by running the relevant python-scripts
