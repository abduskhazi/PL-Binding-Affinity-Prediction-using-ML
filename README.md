# MSc-Project
This repository is maintained for the documentation and coding of the MSc project in Bioinformatics Lab @ Uni-Freiburg.
Please read the file [repo]/report/report.pdf to get an idea of the project.

## Setup
* Please clone the repository. Let this folder be `repo`
* Go inside the `repo` folder. `cd repo`
* Create the required conda environment using the exported conda environment.
`conda env create -f conda_environment/environment.yml`
If the environment is already created, then update it using the command
`conda env update --file conda_environment/environment.yml --prune`
* This creates a conda environment called msc-project. Activate the conda environment.  
`conda activate msc-project`

## Running the model.
* Go to the folder model. `cd model`
* Run any model of your choice. At this time the Random Forest Regressor gives the best result.  
``python random_forest_regresser.py [Execution ID]``  
The Execution ID is optional. Use the same number to reproduce your results.
* The above should create 2 plots plotting *predicted y value* vs *expected y value*.  
`accuracy_validate.png` - Shows how good the model fits on our validation data.  
`accuracy_train.png` - Shows how good the model fits on our training data.

## Reporting Issues / Requesting Features
* Please use [Report issue or request feature](https://github.com/abduskhazi/MSc-Project/issues "Named link title") for this purpose.


## MIT License
Copyright (c) 2021 Abdus Salam Khazi

