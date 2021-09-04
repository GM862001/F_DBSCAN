# F_DBSCAN
## Horizontal and Vertical Federated Versions of DBSCAN
### About
This is the project my Bachelor Thesis was about (3 CFU/ECTS).  
The project consists of the implementation of a horizontal and a vertical federated versions of DBSCAN.  

The whole project is fully illustrated in the file `Federated DBSCAN.pdf`.

#### How to Run the Project
To run the project, follows these steps:
* If you do not have pipenv package installed, install it with the command `pip install pipenv`;
* In the main directory, run the command `pipenv install`, to install the project virtual environment;
* Run the command `pipenv shell` to start a shell in the project virtual environment;
##### Horizontal ederated DBSCAN
* Start the server with the command `python HF_DBSCAN/main_server.py`. This command will run the server on _localhost:8080_.
* Start the clients with the command `python HF_DBSCAN/main_client.py`. This command will run ten clients on _localhost:5000-5009_.
* To start the simulation, connect to _localhost:8080/?action=start_.
* Finally, to check the results of a given client (be it _localhost:5000_), connect to  _localhost:5000/?action=results_.
Note that the script `results/HF_results.py` can be used to automatically start the simulation and retrieve the results. The results will then be scattered and saved as `.png` files in one of the subdirectories of `results.py`, depending on the dataset analyzed.
##### Vertical ederated DBSCAN
* Start the server with the command `python VF_DBSCAN/main_server.py`. This command will run the server on _localhost:8080_.
* Start the clients with the command `python VF_DBSCAN/main_client.py`. This command will run two clients on _localhost:5000-5001_.
* To start the simulation, connect to _localhost:8080/?action=start_.
* Finally, to check the results of a given client (be it _localhost:5000_), connect to  _localhost:5000/?action=results_.
Note that the script `results/VF_results.py` can be used to automatically start the simulation and retrieve the results. The results will then be scattered and saved as `.png` files in one of the subdirectories of `results.py`, depending on the dataset analyzed.
