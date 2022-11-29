# CompEcosystem
Research for CIS 629

# How to run the data imputation algorithm

Syntax: python imputation_script.py --algorithm --columns, --missing_percentage

Algorithm values: knn, mice, missForest, iterativeImputer, ada, extratrees
Column values: 5, 10, 15
Missing_percentage value: 20, 30

python imputation_script.py knn 5 20 > knnOutput.txt
python imputation_script.py mice 5 20 > miceOutput.txt
python imputation_script.py missForest 5 20 > misForestOutput.txt
python imputation_script.py iterativeImputer 5 20 > iterativeImputerOutput.txt
python imputation_script.py ada 5 20 > adaOutput.txt
python imputation_script.py extratrees 5 20 > extratreesOutput.txt
