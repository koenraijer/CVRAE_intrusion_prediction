This code was written as part of the fulfilment of the master's thesis of the Data Science & Society programme at Tilburg University. 

## Recreating the environment
To recreate the environment, paste the following command in the terminal:
```bash
conda create --name <env> --file requirements.txt
```

**NOTE:** Not all code was run on the same machine, e.g. some code was run in an Ubuntu server environment, some code was run locally on MacOS. It is therefore possible that not all of the code in this repository will run as is. 

## Running the code
For an overview of the order in which scripts must be run, including some additional code that was not run within these scripts, please refer to `main.ipynb`. 

## Other relevant code
- Feature generation: `helpers.py` > `prepare_for_ml()`.
- Specific steps during preprecessing: please refer to the commented out with `# FULL PIPELINE` in the first cell of `main.ipynb`. 