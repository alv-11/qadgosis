# SANKOFA
Online Query-aware Dynamic Graph Summaries


## Data
The dataset is available 
[here](https://drive.google.com/drive/folders/1PVlh5KGdQ4qkyEzSnDTCYGmXRuMGwgCT?usp=sharing). 


## C++
The walks component of Algorithm 2 of SANKOFA is implemented in C++ using pybind11. To compile the code, navigate to `kg_summarization/src/core` and run:

`python setup.py clean`

`python setup.py build_ext --inplace`

`python setup.py install`


## Running the code

For a single run of SANKOFA and the baselines, return to the main directory `kg_summarization` and run:

`python3 main.py` 

This reuses the existing query logs.

To generate a new set of queries or compute an average over multiple runs, execute:

`python3 simulate.py`

The output is stored in `multi_runs` with a corresponding folder for each run.