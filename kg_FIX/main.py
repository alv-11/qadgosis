import argparse
import random
import logging
import json
import pdb
import numpy as np
from src.apex import APEX_naive
from src.pagerank import PR
from src.path import DBPEDIA_DATA_DIR, YAGO_DATA_DIR, METAQA_DATA_DIR, METAQA_DATA_DIR_ABLATION, ENCODING
import os
from itertools import chain
import networkx as nx



logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    level=logging.DEBUG,
                    filename = 'log.log', filemode= 'a')

from time import time

from src.base import YAGO, DBPedia, Freebase, MetaQA
from src.user import query_log_by_mids, query_log_by_topics
from src.glimpse import SummaryMethod, GLIMPSE
from src.apex import APEX_naive, APEX, DETAILED_LOGGING
#from src.qadgosis import QADGOSIS
from src.metrics import total_query_log_metrics, average_query_log_metrics

# Available choices for user input arguments in main
# TODO: Change these to point to your local data directories
KG_MAPPING = {
    'YAGO': YAGO(rdf_gz='yagoFacts.gz', query_dir='final/', mid_dir='by-mid/'),
    'Freebase': Freebase(query_dir='queries/final/'),
    'DBPedia': '../PKG_EXP/DBpedia/dbpedia_subgraphs_int', #DBPedia(),
    'MetaQA': MetaQA(ablation=0),
}

METHODS = { 
    'glimpse': SummaryMethod(GLIMPSE, 'GLIMPSE'),
    'glimpse-2': SummaryMethod(GLIMPSE, 'GLIMPSE-2', power=2),
    'apex': SummaryMethod(APEX, 'APEX'),
    'apex-n': SummaryMethod(APEX_naive, 'APEX-N'),
    'pagerank': SummaryMethod(PR, 'PageRank'),
    #'mc': SummaryMethod(QADGOSIS, 'MC')
}



def answer_queries_in_log(KG, K, i, training_set, test_set, summary_methods, acc_list_glimpse, acc_list_apex_n, acc_list_apex, acc_list_pr, update_time_list_glimpse, update_time_apex_n, update_time_apex, update_time_pr, test_size=0.5, query_num_per_test=1):
    """
    :param KG: KnowledgeGraph
    :param K: summary constraint
    :param query_log: list of dict
    :param summary_methods: summarization methods to use
    :param test_size: percent of queries to hold out for testing
    """

    for summary_method in summary_methods:
        logging.info('\t---Summarizing with {}---'.format(summary_method.name()))

        if summary_method.name_[0:4] != 'APEX': # baselines
            if summary_method.name_[0:7] == 'GLIMPSE':
                update_time_list = []
                acc_list = []


                if i==0 :
                    if DETAILED_LOGGING:
                        logging.info('Summarizing for query: {}'.format(i))
                        print("Summarizing for query: ", i)

                    t0 = time()

                    S = summary_method(KG, K, training_set[i]) 

                    runtime = time() - t0
                    print("runtime: ", runtime)
                    update_time_list.append(runtime)



                    total_F1, total_precision, total_recall = total_query_log_metrics(S, test_set[i])
                    if DETAILED_LOGGING:
                        logging.info('\t  Total F1/precision/recall')
                        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                            total_F1, total_precision, total_recall))

                    avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, test_set[i])
                    if DETAILED_LOGGING:
                        logging.info('\t  Average F1/precision/recall')
                        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                            avg_F1, avg_precision, avg_recall))
                    print(total_F1, total_precision, total_recall)
                    print(avg_F1, avg_precision, avg_recall)

                
                else:
                    if DETAILED_LOGGING:
                        logging.info('Re-summarizing for query: {}'.format(i))
                        print("Re-summarizing for query: ", i)

                    t0 = time()
                    


                    def flatten_list(nested_list):
                        """ Recursively flattens a nested list """
                        for item in nested_list:
                            if isinstance(item, list):
                                yield from flatten_list(item)
                            else:
                                yield item
                   

                    keys = list(training_set.keys())
                    target_index = keys.index(i)
                    merged_logs = [training_set[key] for key in keys[:target_index + 1]]
                    merged_logs = list(flatten_list(merged_logs))


                    S = summary_method(KG, K, merged_logs) 

                    runtime = time() - t0
                    print("runtime: ", runtime)
                    update_time_list.append(runtime)



                    print("future queries")
                    total_F1, total_precision, total_recall = total_query_log_metrics(S, test_set[i])
                    if DETAILED_LOGGING:
                        logging.info('\t  Total F1/precision/recall')
                        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                            total_F1, total_precision, total_recall))

                    avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, test_set[i])
                    if DETAILED_LOGGING:
                        logging.info('\t  Average F1/precision/recall')
                        logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                            avg_F1, avg_precision, avg_recall))
                    
                    acc_list.append(avg_F1)
                    print(total_F1, total_precision, total_recall)
                    print(avg_F1, avg_precision, avg_recall)


                    print("sliding window of 2")
                    timesteps = sorted(training_set.keys())

                   #window size = 2
                    window_size = 2
                    for j in range(len(timesteps) - window_size + 1):  # Ensures we don't go out of bounds
                        t1, t2 = timesteps[j : j + window_size]
                        if t2 <= i and i != 0:
                            print(f"Processing window: Timestep {t1} → Timestep {t2}")
                        
                            data_t1 = training_set[t1]
                            data_t2 = training_set[t2]

                            combined_data = data_t1 + data_t2 
                            print(len(combined_data))

                            total_F1, total_precision, total_recall = total_query_log_metrics(S, combined_data)
                            avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, combined_data)


                        # Logging
                            #if DETAILED_LOGGING:
                            #    logging.info(f"\t  [{label}] Total F1/precision/recall: {total_F1:.2f}/{total_precision:.2f}/{total_recall:.2f}")
                            #    logging.info(f"\t  [{label}] Average F1/precision/recall: {avg_F1:.2f}/{avg_precision:.2f}/{avg_recall:.2f}")

                            print(total_F1, total_precision, total_recall)
                            print(avg_F1, avg_precision, avg_recall)

                    # if len(timesteps) >= window_size:
                    #     t1, t2 = timesteps[-window_size:]  
                        
                    #     print(f"Processing window: Timestep {t1} → Timestep {t2}")
                        
                    #     data_t1 = training_set[t1]
                    #     data_t2 = training_set[t2]

                    #     combined_data = data_t1 + data_t2 
                    #     print(len(combined_data))


                    # total_F1, total_precision, total_recall = total_query_log_metrics(S, combined_data )
                    # if DETAILED_LOGGING:
                    #     logging.info('\t  Total F1/precision/recall')
                    #     logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                    #         total_F1, total_precision, total_recall))

                    # avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, combined_data)
                    # if DETAILED_LOGGING:
                    #     logging.info('\t  Average F1/precision/recall')
                    #     logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                    #         avg_F1, avg_precision, avg_recall))
                    
                    acc_list.append(avg_F1)
                    #print(total_F1, total_precision, total_recall)
                    #print(avg_F1, avg_precision, avg_recall)
                    ##    
                    #t+=1

                logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
                logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))

                acc_list_glimpse += acc_list
                update_time_list_glimpse += update_time_list

                np.save(KG.user_dir + '/acc_list_glimpse.npy', acc_list)
                np.save(KG.user_dir + '/time_list_glimpse.npy', update_time_list)
                np.save(KG.user_dir + '/time_list_glimpse.npy', update_time_list)


                


            if summary_method.name_ == 'PageRank':
                update_time_list = []
                acc_list = []
                t = 0
                num_queries_total = len(query_log)
                t0 = time()
                S = summary_method(KG, K, query_log[0:t+1]) # call the object as a function
                runtime = time() - t0
                if DETAILED_LOGGING:
                    logging.info('\t  Time: {:.2f} seconds'.format(runtime))

                # Evaluate question answering
                total_F1, total_precision, total_recall = total_query_log_metrics(S, query_log[1: 1+query_num_per_test])
                if DETAILED_LOGGING:
                    logging.info('\t  Total F1/precision/recall')
                    logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                        total_F1, total_precision, total_recall))

                avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, query_log[1: 1+query_num_per_test])
                if DETAILED_LOGGING:
                    logging.info('\t  Average F1/precision/recall')
                    logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                        avg_F1, avg_precision, avg_recall))
                t+=1
                
                while t < num_queries_total - query_num_per_test:
                    if (t % 1 == 0):
                        if DETAILED_LOGGING:
                            logging.info('Re-summarizing for timestamp: {}'.format(t))
                            print("Re-summarizing for timestamp: ", t)

                        t0 = time()
                        S = summary_method(KG, K, query_log[0:t+1]) # call the object as a function
                        runtime = time() - t0
                        update_time_list.append(runtime)

                        total_F1, total_precision, total_recall = total_query_log_metrics(S, query_log[t+1: t+1+query_num_per_test])
                        if DETAILED_LOGGING:
                            logging.info('\t  Total F1/precision/recall')
                            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                                total_F1, total_precision, total_recall))

                        avg_F1, avg_precision, avg_recall = average_query_log_metrics(S, query_log[t+1: t+1+query_num_per_test])
                        if DETAILED_LOGGING:
                            logging.info('\t  Average F1/precision/recall')
                            logging.info('\t    {:.2f}/{:.2f}/{:.2f}'.format(
                                avg_F1, avg_precision, avg_recall))
                        
                        acc_list.append(avg_F1)
                        
                    t+=1

                logging.info('\t  Ave Time on Each Training Log: {:.2f} seconds'.format(np.mean(update_time_list)))
                logging.info('\t  Ave Ave F1 on Each Training Log: {:.2f}'.format(np.mean(acc_list)))

                acc_list_pr += acc_list
                update_time_pr += update_time_list


                np.save(KG.user_dir + '/acc_list_pr.npy', acc_list)
                np.save(KG.user_dir + '/time_list_pr.npy', update_time_list)


                
            
        else:
            # APEX
            if summary_method.name_ == 'APEX-N':
                acc_list, update_time_list = APEX_naive(KG, K, query_log, query_num_per_test)
                acc_list_apex_n += acc_list
                update_time_apex_n += update_time_list
            if summary_method.name_ == 'APEX': 
                acc_list, update_time_list = APEX(KG, K, query_log, query_num_per_test)
                acc_list_apex += acc_list
                update_time_apex += update_time_list



def float_in_zero_one(value):
    """Check if a float value is in [0, 1]"""
    value = float(value)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError('Value must be a float between 0 and 1')
    return value

def positive_int(value):
    """Check if an integer value is positive"""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Value must be positive')
    return value

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--kg', choices=list(KG_MAPPING.keys()), default='YAGO',
            help='KG to summarize')
    parser.add_argument('--n-queries', type=positive_int, default=100,
            help='Number of queries to simulate per user. Default is 100.')
    parser.add_argument('--n-topic-mids', type=positive_int, default=2,
            help='Number of topic mids of interest per user. Default is 2.')
    parser.add_argument('--n-topics', type=positive_int, default=3,
            help='Number of topics to simulate per user log. '
                 'For Freebase only. Default is 3.')
    parser.add_argument('--n-mids-per-topic', type=positive_int, default=20,
            help='Number of unique MIDs per topic. For Freebase only. Default is 20.')
    parser.add_argument('--n_users', type=positive_int, default=2,
            help='Number of users to simulate. Default is 1.')
    parser.add_argument('--start_user', type=positive_int, default=0,
            help='Number of users to simulate. Default is 0.')
    parser.add_argument('--test-size', type=float_in_zero_one, default=0.5,
            help='Percentage of queries per user to hold out for testing, '
                 'in [0, 1]. Default is 0.5.')
    parser.add_argument('--percent-triples', type=float_in_zero_one, default=0.001,
            help='Ratio of number of triples of KG to use as K '
                 '(summary constraint). Default is 0.001.')
    parser.add_argument('--random-query-prob', type=float_in_zero_one, default=0,
            help='Probability of users asking random queries rather '
                 'than topic-specific ones. Default is 0.')
    parser.add_argument('--shuffle', action='store_true',
            help='Set this flag to true to shuffle all generated logs. Default False.')
    # parser.add_argument('--method', nargs='+', default=['glimpse'],
    #         choices=list(METHODS.keys()),
    #         help='Summarization methods to call. Default is [glimpse].')
    parser.add_argument('--method', nargs='+', default=['apex'],
            choices=list(METHODS.keys()),   # glimpse, glimpse-2, apex, apex-n
            help='Summarization methods to call. Default is [apex].')
    parser.add_argument('--save-queries', action='store_true', 
            help='Whether to save the generated queries (warning: overwrite existing saves). Default False')
    parser.add_argument('--load-queries', action='store_true', 
            help='Whether to load queries from dataset folder (warning: make sure dataset folder is non-empty, also make sure enough amount of queries than acquired). Default False and default is to generate new queries')
    parser.add_argument('--query-num-per-test', type=positive_int, default=1,
            help='Number of users to simulate. Default is 1.')
    
    return parser.parse_args()

def main():
    acc_list_glimpse = []
    acc_list_apex_n = []
    acc_list_apex = []
    acc_list_pr = []
    update_time_list_glimpse = []
    update_time_apex_n = []
    update_time_apex = []
    update_time_pr = []
    args = parse_args()
    if (args.save_queries is True) and (args.load_queries is True):
        raise NotImplementedError

    if (args.save_queries is True):
        print('Warning: File Rewrite, proceed?')
        pdb.set_trace()

    if args.kg == 'DBPedia':
        dbpedia_dir = KG_MAPPING[args.kg]
        for user in range(args.start_user, args.start_user + args.n_users):
            logging.info('---Simulating user {}---'.format(user))
            logs = dict()
            dynamic_graph = dict()
            for i, file_name in enumerate(sorted(os.listdir(dbpedia_dir))):
                graph = os.path.join(dbpedia_dir, file_name)           
                print(graph)
                KG = DBPedia(rdf_gz=graph)
                summary_methods = [METHODS[name] for name in args.method]

                # Load the KG into memory
                logging.info('Loading {}'.format(KG.name()))
                KG.load()
                logging.info('Loaded {}'.format(KG.name()))

                #convert into directed graph for monte carlo method
                directed_graph = nx.DiGraph()

                edges = KG.triples()
                for s, p, o in edges:
                    directed_graph.add_edge(s, o, predicate=p)

                    directed_graph[s][o]['weight'] = 0.01

                dynamic_graph[i] = directed_graph


                # Number of triples for summary
                #K = int(args.percent_triples * KG.number_of_triples())
                K = 200
                logging.info('K = {}'.format(K))
                KG.update_user(user)

                #logging.basicConfig(filename=KG.user_dir + '/log.log')

                if args.kg == 'Freebase':
                    topics = random.sample(KG.topics(), k=args.n_topics)

                    query_log = query_log_by_topics(
                            KG, topics, args.n_mids_per_topic, args.n_queries,
                            shuffle=args.shuffle, random_query_prob=args.random_query_prob)
                
                elif args.kg == 'MetaQA':
                    args.save_queries = False
                    args.load_queries = True
                    query_log = []
                    for i in range(args.n_queries):
                        with open(KG.query_dir() + "q"+ str(i)+ ".json", "r") as f:
                            query_log.append(json.load(f))

                    logging.info('---Loaded a log of {} queries----'.format(len(query_log)))
                

                else:
                    # topic_mids = random.sample(KG.topic_mids(), k=args.n_topic_mids)
                    topic_mids = random.sample(list(KG.triples_.keys()), k=args.n_topic_mids)

                    if args.load_queries:
                        query_log = []
                        for i in range(args.n_queries):
                            with open(KG.query_dir() + "q"+ str(i)+ ".json", "r") as f:
                                query_log.append(json.load(f))

                        logging.info('---Loaded a log of {} queries----'.format(len(query_log)))

                    else:

                        query_log = query_log_by_mids(
                                KG, topic_mids, args.n_queries, topic_dist=np.ones(len(topic_mids))/len(topic_mids),
                                shuffle=args.shuffle,
                                random_query_prob=args.random_query_prob, whether_save = args.save_queries)
                        logging.info('---Generated a log of {} queries----'.format(len(query_log)))

                logs[i] = query_log
    
                save_dir = KG.query_dir()
                # if args.save_queries:
                #     logging.info('---Saving generated queries---')
                #     logs[i] = query_log
                    #print(query_log)
                training_data = dict()
                test_data = dict()

                for i,j in logs.items():
                    log = j 
                    random.shuffle(log)
                    len_logs = len(log)
                    train_size = int(0.8* len_logs)  

                    training_set = log[:train_size]
                    test_set = log[train_size:]
                    
                    training_data[i] = training_set
                    test_data[i] = test_set
                    #print(len(training_set), len(test_set), len(log))


                answer_queries_in_log(KG, K, i, training_data, test_data, summary_methods, acc_list_glimpse, acc_list_apex_n, acc_list_apex, acc_list_pr, update_time_list_glimpse, update_time_apex_n, update_time_apex, update_time_pr, test_size=args.test_size, query_num_per_test=args.query_num_per_test)

            if args.save_queries:
                logging.info('---Saving generated queries---')        
                for graph, log_ in logs.items():
                    file_path = os.path.join(save_dir, f"q{graph}.json")
                    print(file_path)

                    #if not os.path.exists(file_path):
                    with open(file_path, "a") as outfile:
                        print(f"Saving query log {graph} to {file_path}")  # Debug print
                        json.dump(log_, outfile, indent=4)

            #print(logs.keys())
            # for graph, query_log in logs.items():
            #     file_path = os.path.join(KG.query_dir(), f"q{graph}.json")
            #     print(file_path)
                
            #     #if not os.path.exists(file_path):
            #     with open(file_path, "a") as outfile:
            #         print(f"Saving query log {graph} to {file_path}")  # Debug print
            #         json.dump(query_log, outfile, indent=4)
                #else:
                   # print(f"File {file_path} already exists! Skipping...")

                        
                    #for i in range(len(query_log)):
                
                    # with open(KG.query_dir() + "q"+ str(i)+ ".json", "w") as outfile:
                    #     print(KG.query_dir()+ "q"+ str(i)+ ".json")
                    #     json.dump(query_log, outfile, indent=4)

        #answer_queries_in_log(KG, K, query_log, summary_methods, acc_list_glimpse, acc_list_apex_n, acc_list_apex, acc_list_pr, update_time_list_glimpse, update_time_apex_n, update_time_apex, update_time_pr, test_size=args.test_size, query_num_per_test=args.query_num_per_test)

    # save arrays
    DATA_DIRS = {'YAGO': YAGO_DATA_DIR, 'DBPedia': DBPEDIA_DATA_DIR, 'MetaQA': METAQA_DATA_DIR}
    DATA_DIR = DATA_DIRS[args.kg]
    if len(acc_list_glimpse) > 0:
        np.save(DATA_DIR + '/acc_list_glimpse.npy', acc_list_glimpse)
    if len(acc_list_apex_n) > 0:
        np.save(DATA_DIR + '/acc_list_apex_n.npy', acc_list_apex_n)
    if len(acc_list_apex) > 0:
        np.save(DATA_DIR + '/acc_list_apex.npy', acc_list_apex)
    if len(acc_list_pr) > 0:
        np.save(DATA_DIR + '/acc_list_pr.npy', acc_list_pr)
    if len(update_time_list_glimpse) > 0:
        np.save(DATA_DIR + '/update_time_list_glimpse.npy', update_time_list_glimpse)
    if len(update_time_apex_n) > 0:
        np.save(DATA_DIR + '/update_time_apex_n.npy', update_time_apex_n)
    if len(update_time_apex) > 0:
        np.save(DATA_DIR + '/update_time_apex.npy', update_time_apex)
    if len(update_time_pr) > 0:
        np.save(DATA_DIR + '/update_time_pr.npy', update_time_pr)

    logging.info('Shutting down...')
if __name__ == '__main__':
    main()