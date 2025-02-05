# APEX_submission

## file structure
-APEX_submission
    -code
        -src (utils)
        -main.py (run this, specific commands will be provided below)
        -README.md (this file)
    -PKG_EXP
        -DBpedia
            -queries (query folder)
                -user # (user folder)
            -facts.gz (dataset)
        -MetaQA    
            -1-hop, 2-hop, 3-hop (initial MetaQA dataset to generate queries)
            kb.txt (dataset)
            -dataprocessing.py (utils when generating queries)
            -generate_query.py (can be used to re-generate queries)
            -queries (query folder)
                -user # (user folder)
        -YAGO
            -queries (empty, will be generated)
            -*missing large dataset file, need to be manually downloaded. Instructions in both paper and below* yagoFacts.gz


## Datasets
We provide DBpedia and MetaQA that we used in our experiments along with the source code. YAGO has to be manually downloaded from https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago/downloads/. The button is Download YAGO Themes -> CORE -> yagoFacts -> Download TSV. Convert to ".gz" file using gzip if not in this format.

The ablation study uses the same MetaQA dataset. If you want to reproduce the experiment result, simply copy-paste MetaQA folder and change its name to MetaQA_ablation, after that revise METAQA_DATA_DIR_ABLATION in code/src/path.py file.


## How to run
Go to code/src/path.py, change the dataset path if needed. The resulted log file name can be set in main.py line 14.

For each dataset, you can run the corresponding command. We recommand to run MetaQA first because DBPedia and YAGO are very large knowledge graph datasets and takes time to finish executing.

'''
cd code

python main.py --kg MetaQA --method glimpse pagerank apex-n apex --n-queries 200 --n-topic-mids 20 --percent-triples 0.0001 --n_users 10

python main.py --kg DBPedia --method glimpse pagerank apex-n apex --save-queries --n-queries 200 --n-topic-mids 20 --percent-triples 0.000001 --n_users 10

python main.py --kg YAGO --method glimpse pagerank apex-n apex --save-queries --n-queries 200 --n-topic-mids 20 --percent-triples 0.000001 --n_users 10
'''

If you want to use the same queries stored, change --save-queries to --load-queries. When run with --save-queries, the query files will be re-written, you need to enter "c" when pdb askes whether to proceed.


## Result
For convenience, we save the experiment results in the log files. We record F1 score and running time for each adapting. To quickly view the result, search "Ave Time on Each Training Log" and "Ave Ave F1 on Each Training Log" in the log files. Note that the "Summarizing with ......" under "Ave Ave F1 on Each Training Log" is not a pair. It's the next method applied after the method that generated this result.