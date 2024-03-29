# F_DBSCAN
## Implementation and Analysis of Horizontal and Vertical Federated Versions of DBSCAN
### About
This my Bachelor Thesis project.  
The project consists of the implementation of a horizontal and a vertical federated versions of DBSCAN.  
The project is introduced in the file `Presentazione.pdf` (written in Italian) and fully described in the file `Federated DBSCAN.pdf`.  
#### How to Run the Project
To run the project, follows these steps:
* If you do not have pipenv package installed, install it with the command `pip install pipenv`;
* In the main directory, run the command `pipenv install`, to install the project virtual environment;
* Run the command `pipenv shell` to start a shell in the project virtual environment;
##### Horizontal federated DBSCAN
* Enter the sub-project directory `HF_DBSCAN/`;
* Start the server with the command `python main_server.py`. This command will run the server on _localhost:8080_;
* Start the clients with the command `python main_client.py`. This command will run ten clients on _localhost:5000-5009_;
* To start the simulation, connect to _localhost:8080/?action=start_;
* Finally, to check the results of a given client (be it _localhost:5000_), connect to  _localhost:5000/?action=results_.
Note that the script `results/HF_results.py` can be used to automatically start the simulation and retrieve the results. To run the script, first enter the directory `results/`, and then run the command `python HF_results.py`. The results will then be scattered and saved as `.png` files in one of the subdirectories of `results/`, depending on the dataset analyzed. Furthermore, some statistics about the algorithm efficiency will be printed.
##### Vertical federated DBSCAN
* Enter the sub-project directory `VF_DBSCAN/`;
* Start the server with the command `python main_server.py`. This command will run the server on _localhost:8080_.
* Start the clients with the command `python main_client.py`. This command will run two clients on _localhost:5000-5001_.
* To start the simulation, connect to _localhost:8080/?action=start_.
* Finally, to check the results of a given client (be it _localhost:5000_), connect to  _localhost:5000/?action=results_.
Note that the script `results/VF_results.py` can be used to automatically start the simulation and retrieve the results. To run the script, first enter the directory `results/`, and then run the command `python VF_results.py`. The results will then be scattered and saved as `.png` files in one of the subdirectories of `results/`, depending on the dataset analyzed. Furthermore, some statistics about the algorithm efficiency will be printed.
