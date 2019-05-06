import os

if 'implicit_corpus' not in os.environ:
    raise KeyError("Please supply the directory as environment "
                   "variable: 'implicit_corpus'")
else:
    base = os.environ['implicit_corpus']

c.HashParam.event_vocab = os.path.join(
    base, 'gigaword_events', 'embeddings', 'event_embeddings_mixed.voc')
c.HashParam.word_vocab = os.path.join(
    base, 'gigaword_word_embeddings', 'word_embeddings.voc')
c.HashParam.frame_dep_map = os.path.join(
    base, 'frame_maps', 'frames_args_filled.tsv')
c.HashParam.dep_frame_map = os.path.join(
    base, 'frame_maps', 'args_frames_filled.tsv')
c.HashParam.component_vocab_dir = os.path.join(
    base, 'gigaword_events', 'vocab/')
c.HashParam.frame_files = '/home/zhengzhl/resources/fndata-1.5/frame'
