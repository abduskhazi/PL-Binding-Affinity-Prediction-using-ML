# MSc-Project
This repository is maintained for the documentation and coding of the MSc project.
Please read the weekly notes committed to get an idea about the work done till now in the project.

## Setup
* Please clone the repository. Let this folder be `repo`
* Go inside the `repo` folder. `cd repo`
* Create the required conda environment using the exported conda environmentt.  
`conda env create -f conda_environment/environment.yml`
* This creates a conda environment called msc-project. Activate the conda environment.  
`conda activate msc-project`

## Running the model.
* Go to the folder model. `cd model`
* Run any model of your choice. At this time the Random Forest Regressor gives the best result.  
``python random_forest_regresser.py``
* The above should create 2 plots plotting *predicted y value* vs *expected y value*.  
`temp_validate.png` - Shows how good the model fits on our validation data.  
`temp_train.png` - Shows how good the model fits on our test data.

## Reporting Issues / Requesting Features
* Please use [Report issue or request feature](https://github.com/abduskhazi/MSc-Project/issues "Named link title") for this purpose.
