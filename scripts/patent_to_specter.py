import numpy as np
import pandas as pd
import scipy 
import logging

import sys
import glob
import json
import datetime
import argparse

import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
#sys.path.append("/Users/kipnisal/AI_Pilot/Code/")
from Patent import Patent


def load_patents(path_to_folder, lemmatize=False) :
    lo_files = glob.glob(path_to_folder + '/*')
    #glob.glob(path_to_folder + '/*.XML') \
             #+ glob.glob(path_to_folder + '/*.xml')
    lo_patents = []
    logging.info(f"Reading patent applications from {path_to_folder}...")
    for fn in tqdm(lo_files) : 
      try :
        pt = Patent(fn)
        if lemmatize :
          pt.lemmatize_tree()
        lo_patents += [pt]
      except :
        logging.warning("Could not read Patent information from"
                        " file {}.".format(fn))

    logging.info(f"Found {len(lo_patents)} items.")
    return dict([(pt.get_id(), pt)  for pt in lo_patents])


def get_topic_connections(lo_patents) :
    """
    Lists all applications sahring the same list of topics as topic-connected


    To do: Soften connectivity criterion to perfect match between the lists
    """
    def where_in_list(ls) :
        return df[pd.DataFrame(df.cpc.tolist()).isin(ls).any(1)].doc_id.tolist()

    pat = "^[a-z][0-9][0-9] *[a-z]\s*[0-9]{1,2}( |/)*[0-9]{0,4}"
    prog = re.compile(pat)

    df = pd.DataFrame()

    for pt_id in tqdm(lo_patents) :
        pt = lo_patents[pt_id]
        metadata = {}
        metadata['doc_id'] = pt.get_id()
        metadata['cpc'] = [re.sub("[ /]","",prog.match(code.lower()).group()) for code in pt.get_classification_code()]
        df = df.append(metadata, ignore_index = True)

    df.loc[:,'ref'] = df['cpc'].apply(where_in_list)
    df.apply(lambda row : row['ref'].remove(row['doc_id']), axis=1)

    df.loc[:,'no_ref'] = df.ref.apply(lambda x : len(x))

    return df

def get_citation_info(lo_patents, use_topic_ref = False) :

    def fetch_id(pt_id) :
        raw_id = re.findall('([0-9]+)-', pt_id)[0]
        if lo_patents.get(raw_id) :
            return raw_id
        else :
            return raw_id+'(missing)'
        
    dfr = pd.DataFrame()
    logging.info("Reading rejection information...")
    for pt_id in tqdm(lo_patents) :
        pt = lo_patents[pt_id]
        metadata = {}
        metadata['doc_id'] = pt_id
        lo_rej = pt.get_rejections()
        rej_ids = list(set([fetch_id(rid) for rid in lo_rej]))
        metadata['ref'] = rej_ids
        metadata['existing_ref'] = [t for t in rej_ids if 'missing' not in t]
        dfr = dfr.append(metadata, ignore_index = True)

    # back ref:

    dfr = dfr.set_index('doc_id')
    
    if use_topic_ref  :
        logging.info("Reading topic information...")
        df_topic = get_topic_connections(lo_patents)
        dfr = dfr.join(df_topic[['doc_id', 'ref', 'no_ref']].set_index('doc_id'), rsuffix='_topic')
    else :
        dfr.loc[:, 'ref_topic'] = dfr.apply(lambda _: [], axis=1)
        dfr.loc[:, 'no_ref_topic'] = dfr.apply(lambda _: [], axis=1)
        
    dfr['back_ref'] = dfr.apply(lambda _: [], axis=1)
    for row in dfr.iterrows() :
        for r in row[1]['existing_ref'] :
            dfr.at[r, 'back_ref'] += [r]
                        
    dfr.loc[:, 'no_ref'] = dfr.ref.apply(lambda x : len(x))
    dfr.loc[:, 'no_existing_ref'] = dfr.existing_ref.apply(lambda x : len(x))
    dfr.loc[:, 'no_back_ref'] = dfr.back_ref.apply(lambda x : len(x))
    
    dfr = dfr.reset_index()
    return dfr.rename(columns = {'existing_ref' : 'ref_strong',
                 'back_ref' : 'ref_weak', 'ref_topic' : 'ref_neg',
                                'no_ref_topic' : 'no_ref_neg'})


def main() :
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='lemmatize patents')
    parser.add_argument('-i', type=str, help='path to data folder')
    parser.add_argument('-o', type=str, help='output folder')

    parser.add_argument("--use-topics", default=False, action="store_true" ,
                                             help="use topic information"
                                             "to infere negative citatino"
                                             )
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args() 
    use_topics = args.use_topics
    in_folder = args.i
    out_folder = args.o

    # +=========================
    # | script to generate files readable by SPECTER from rejection and topic 
    # | information.
    # +========================

    lo_patents = load_patents(in_folder)

    df = get_citation_info(lo_patents, use_topic_ref=use_topics)
    print("Total # of references: ", df.no_ref.sum())
    print("Total # of existing references: ", df.no_existing_ref.sum())
    print("Total # of back references: ", df.no_back_ref.sum(), " (sanity check)")
    print("Total # of negative references: ", df.no_neg_ref.sum())

    res = {}
    for r in df.iterrows() :
        dc_neg = dict([(idd, {'count' : 1}) for idd in r[1]['ref_neg']])
        dc_weak = dict([(idd, {'count' : 4}) for idd in r[1]['ref_weak']])
        dc_strong = dict([(idd, {'count' : 5}) for idd in r[1]['ref_strong']])

        dc = {}
        dc.update(dc_neg)
        dc.update(dc_weak)
        dc.update(dc_strong) # Note: updating order is important!
        if dc != {} :
            res[r[1]['doc_id']] = dc

    # Serializing json    
    fn1 = out_folder + "/" + "data.json"
    with open(fn1, "w") as f :
        json_object = json.dump(res, f, indent = 2)
    print("saved citation information in ",fn1)
        
    # save train, val, test
    val_size = 10
    test_size = 10

    val_idcs = np.random.choice(list(df.index), val_size)
    val = df.loc[val_idcs,:].doc_id.tolist()
    with open(out_folder + "/" + 'val.txt','wt') as f :
        for doc in val :
            f.write(doc)
            f.write('\n')

    test_idcs = np.random.choice(list(df.index), test_size)
    test = df.loc[test_idcs,:].doc_id.tolist()
    with open(out_folder + "/" + 'test.txt','wt') as f :
        for doc in test :
            f.write(doc)
            f.write('\n')

    train_idcs = [idc for idc in list(df.index) if idc not in (test_idcs + val_idcs)]
    train = df.loc[train_idcs, :].doc_id.tolist()
    with open(out_folder + "/" + 'train.txt','wt') as f :
        for doc in train :
            f.write(doc)
            f.write('\n')
            
    def english_only(text) :
        return re.sub("[^a-zA-Z0-9 ,-=;.:,]","", text)

    df = pd.DataFrame()
    for pt_id in lo_patents :
        pt = lo_patents[pt_id]
        df = df.append({'paper_id' : english_only(pt.get_id()),
                       'title' : english_only(pt.get_title()),
                       'abstract' : english_only(pt.get_claims()['claim1'])},
                      ignore_index=True)    

    res = {}
    for r in df.set_index('paper_id').iterrows() :
        res[r[0]] = {'title' : r[1]['title'], 'abstract' : r[1]['abstract'], 'paper_id' : r[0]}

    # Serializing json    
    fn2 = out_folder + "/" + "metadata.json"
    with open(fn2, "w") as f :
        json_object = json.dump(res, f, indent = 2)
    print("saved data fields information in ",fn2)

if __name__ == '__main__':
    main()