# qadgosis
Complete Robust Online Graph Summaries

I created new directories for the code so I don't mix things up. `original` is the original code on the synthetic graph, `kg_python` is the code applied to KGs and `kg_cplusplus` is the C++ code applied to KGs.

To run the code for QADGOSIS, run the following command:

### 1.  original 
`python3 original/main.py`

The output will be saved in the `output` folder and it includes the logs and the details of the results.


### 2. kg_python
Navigate to the directory `kg_python`

To download the dataset run: `python3 src/util.py`
It will download the dynamic dataset and store it in the `PKG_EXP` folder. 

To run for QADGOSIS and PageRank, run: `python3 montecarlo.py`

Originally adapted from the APEX paper, for GLIMPSE run:

`python main.py --kg DBPedia --method glimpse --save-queries --n-queries 20 --n-topic-mids 4 --n_users 1`

It can be udpated later to also compare with APEX and PageRank

If you want to use the same queries stored, change --save-queries to --load-queries. When run with --save-queries, the query files will be re-written, you need to enter "c" when pdb askes whether to proceed.

I have fixed the summary size for GLIMPSE at 200 based on the average number of triples obtained for TOP-100 QADGOSIS and PageRank.

### 3. kg_cplusplus
To create  an int version of the graph which is required for the C++ code, run:
`python3 src/util.py`

To compile the C++ code, navigate to `src/qadgsis` and run:

`python setup.py clean`

`python setup.py build_ext --inplace`

`python setup.py install`


Finally return to the main directory `kg_cplusplus` and run:

`python3 montecarlo.py`