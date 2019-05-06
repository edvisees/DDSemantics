# Implicit Argument Setup
Experiments setups:
1. basic
1. basic_arg_comp_3: using 3 layers for arg composition instead of 2
1. basic_event_comp_3: using 3 layers for event composition instead of 2
1. basic_gaussian_distance: use gaussian to simulate distances
1. basic_biaffine:

Preprocessing training dataset:
1. Parse the large dataset with Stanford and Semafor (for example, get nyt_events.json.gz)
1. Split it into sub files and shuffle the order
    1. ```gunzip -c nyt_all_frames.json.gz | split -l 50000 - nyt_frames_shuffled/part_  --filter='shuf | gzip > $FILE.gz```
1. Calculate vocabulary
    1. ```python -m event.arguments.prepare.event_vocab --input_data /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/nyt_events.json.gz --vocab_dir /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/vocab --embedding_dir /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/embeddings --sent_out /media/hdd/hdd0/data/arguments/implicit/gigaword_corpus/```
    1. This will also create event sentences to train embeddings
1. Creat frame_mapping
    1. ```create_frame_mappings.sh```
        1. ```python -m event.arguments.prepare.frame_collector nyt_events.json.gz frame_maps```
        1. ```python -m event.arguments.prepare.frame_mapper frame_map```
1. Train event embedding
    1. ```scripts/get_sents_emb_all_frames.sh```
    1. ```scripts/get_sents_emb_event_only.sh```
1. Hash the dataset
    1. ```scripts/hash_train_event_only.sh```
    1. ```scripts/hash_train_all_frames.sh```

Test set setups:
1. Create the automatically constructed training set
    1. Find a domain relevant corpus and parse it with the pipeline
    1. Use the pre-parsed annotated Gigaword NYT portion
1. Obtain the relevant corpus
    1. For G&C Corpus
        1. Read both Propbank and Nombank into the annotations format
        1. Add G&C data into the dataset        
    1. For SemEval2010 Task 10
        1. Read SemEval dataset into the annotation format
        1. ```python -m event.io.dataset.reader negra```
    1. Run ImplicitFeatureExtractionPipeline to create dataset with features.
        1. Now you will get cloze.json.gz
    1. Run hasher to convert it to the Integer format
        1. Use different conf will use different vocab (filter or not) 
        1. ```python -m event.arguments.prepare.hash_cloze_data conf/implicit/hash.py --HashParam.raw_data=cloze.json.gz --HashParam.output_path=cloze_hashed.json.gz```
        1. Or simply ```scripts/hash_test_data.sh```