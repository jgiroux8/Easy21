# Easy21
Reinforcement Learning on the Easy21 environment with Sarsa and Monte-Carlo Control.


The code base functions off a configuration file (config.json), which specifies the following fields:


* "gamma": 1.0,
* "lambda\_values": [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
* "episodes": 500000,
* "run\_MC": 1,
* "run\_sarsa": 1

The "run" fields are Boolean flags to run specific algorithms. If the MC algorithm is not run, only certain Sarsa plots will be generated. The code will run Sarsa($\lambda$) for all values specified. Generated figures will be placed in the newly created Figures folder. If the folder already exists, \textit{i.e.}, you have run the code before, it will overwrite existing files or create new ones in the folder depending on the configuration. Once the configuration file has been set, the code can be run with the following command:


python run.py --config config.json


