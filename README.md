# SANKOFA
Query-aware Online Incremental Graph Summaries

I created new directories for the code so I don't mix things up. `original` is the original code on the synthetic graph, `kg_python` is the code applied to KGs and `kg_cplusplus` is the C++ code applied to KGs.

To run the code for SANKOFA, run the following command:


### 3. kg_cplusplus
To create  an int version of the graph which is required for the C++ code, run:
`python3 src/util.py`

To compile the C++ code, navigate to `src/qadgsis` and run:

`python setup.py clean`

`python setup.py build_ext --inplace`

`python setup.py install`


Finally return to the main directory `kg_cplusplus` and run:

`python3 montecarlo.py`

output for simulate.py = multi_runs with a spearte direcoty for each run