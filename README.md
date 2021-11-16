# MSc-Project
This repository is maintained for the documentation and coding of the MSc project.
Please read the weekly notes committed to get an idea about the work done till now in the project.

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

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
