# DINOBot PyBullet Implementation

This repository contains the PyBullet implementation of DINOBot, developed as part of my Master's thesis. It contains
code used to perform DINOBot on a number of different objects, record demonstrations, attempt transfers between them,
and many more. It contains a separate DINO server module which allows for running the DINO computations on the GPU
cluster and using the GUI locally. 

It also contains the data generated as a part of this project, with 51 object demonstrations, thousands of transfer
attempts, and additional training data.

More information about the implemented methods, including a user guide, can be found in the dissertation report. 

This repository was private for the duration of the project and made public on the 5th of September 2024.

### Structure
The repository contains the following:
 - `_generated` - The generated artefacts, including the transfer attempts and success rates
 - `additional_urdfs` - Additional URDF models used
 - `demonstrations` - Demonstrations for 51 objects + additional 26 for training
 - `DINOserver` - The server module
   - `client.py` - The client side which sends requests
   - `correspondences.py` - For calculating correspondences
   - `dinobot_utils.py` - For speeding up the correspondence finding 
   - `requirements.txt` - The requirements file
   - `server.py` - The server side which receives requests and sends the calculated correspondences
 - `tests` - A mixture of manual and automated tests for the different parts of the pipeline
 - `config.py` - The configuration used throughout the repository
 - `connerctor.py` - The connector between the DINOBot code and the server
 - `database.py` - Code related to database creation and operations
 - `demo_sim_env.py` - For recording demonstrations using keyboard control
 - `demo_transfer.py` - For running different transfer experiments
 - `dinobot.py` - The main DINOBot implementation
 - `helper.py` - Various helper methods
 - `hyper_runner.sh` - The runner for hyperparameter search
 - `hyperparameter_search.py` - The code for hyperparameter search for DINOBot using wandb
 - `requirements.txt` - The main requirements file
 - `server_runner.sh` - The runner for the server 
 - `sim_debug_demo_capturing.py` - Debug controlled demonstration recording
 - `sim_env.py` - The main simulation environment that controls all the movement and such
 - `task_types.py` - An enum defining the task types
 - `transfer_runner.sh` - A runner for the transfer experiments

