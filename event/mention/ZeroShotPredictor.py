from event.io.dataset.base import (
    Span,
    DataLoader,
    DEDocument,
    Corpus,
)
import json
import os
import sys
import tempfile
from event.arguments.prepare.event_vocab import EmbbedingVocab

from traitlets.config import Configurable
from traitlets import (
    Int,
    List,
    Unicode,
)
from event.util import load_mixed_configs, set_basic_log

import logging
import numpy as np
from collections import defaultdict
import pprint
from scipy.spatial.distance import cosine
from event.mention import aida_maps
import traceback


class ZeroShotEventResources(Configurable):
    """
    Resource class.
    """
    event_embedding_path = Unicode(help='Event Embedding path').tag(config=True)
    word_embedding_path = Unicode(help='Word Embedding path').tag(config=True)

    event_vocab_path = Unicode(help='Event Vocab').tag(config=True)
    word_vocab_path = Unicode(help='Word Vocab').tag(config=True)

    target_ontology = Unicode(help='Ontology path').tag(config=True)

    def __init__(self, **kwargs):
        super(ZeroShotEventResources, self).__init__(**kwargs)

        self.event_embedding = np.load(self.event_embedding_path)
        self.word_embedding = np.load(self.word_embedding_path)

        self.event_embed_vocab = EmbbedingVocab(self.event_vocab_path)
        self.word_embed_vocab = EmbbedingVocab(self.word_vocab_path)

        with open(self.target_ontology) as onto_file:
            self.onto_set = set()

            self.ontology = json.load(onto_file)

            for frame in self.ontology['frames']:
                self.onto_set.add(frame['@id'])

        logging.info(
            f"{len(self.event_embed_vocab.vocab)} events in embedding.")

        logging.info(
            f"{len(self.word_embed_vocab.vocab)} words in embedding."
        )


def camel_slash_split(s, lower=True):
    l_s = [[]]

    for c in s:
        if c.isupper():
            l_s.append([])

        if c == '_':
            l_s.append([])
        else:
            l_s[-1].append(c)

    if lower:
        return [''.join(l).lower() for l in l_s if len(l) > 0]
    else:
        return [''.join(l) for l in l_s]


def event_type_split(s):
    if s in aida_maps.kbp_type_split_map:
        return aida_maps.kbp_type_split_map[s]
    else:
        return camel_slash_split(s)


class ZeroShotTypeMapper:
    def __init__(self, resources):
        self.resources = resources
        # A map from the ontology event type to its tokens
        self.onto_event_tokens = {}
        self.onto_arg_role_tokens = {}

        self.onto_arg_domain = defaultdict(list)

        # From a fine-grained to a parent
        self.type_parent = {}
        # The embedding of the predicates
        self.pred_embeds = {}

        self.tokenize_ontology()

    def tokenize_ontology(self):
        for frame in self.resources.ontology['frames']:
            onto_category = frame['@type']
            if onto_category == 'event_type':
                event_type = frame['@id']
                self.onto_event_tokens[event_type] = {}
                level_types = event_type.split(':')[1].split('.')

                tokenized_types = []
                for lt in level_types:
                    tokenized_types.append([])
                    for t in camel_slash_split(lt):
                        if t not in aida_maps.ldc_ontology_skips:
                            t = aida_maps.onto_token_nom_map.get(t, t)
                            pred_id = self.resources.event_embed_vocab \
                                .get_index(t + '-pred', None)
                            if pred_id >= 0:
                                tokenized_types[-1].append(t)
                                pred_vector = self.resources.event_embedding[
                                    pred_id]
                                self.pred_embeds[t] = pred_vector
                            else:
                                logging.debug(f"Predicate form for {t} not "
                                              f"found")

                self.onto_event_tokens[event_type]['top'] = tokenized_types[0]
                if len(level_types) > 2:
                    self.onto_event_tokens[event_type]['middle'] = \
                        tokenized_types[1]
                    self.onto_event_tokens[event_type]['low'] = tokenized_types[
                        2]

                    if not frame['subClassOf'] == 'aida:EventType':
                        self.type_parent[event_type] = frame['subClassOf']

            if onto_category == 'event_argument_role_type':
                role_type = frame['@id']
                role_token = role_type.split('_')[-1]
                self.onto_arg_role_tokens[role_type] = [role_token.lower()]
                self.onto_arg_domain[frame['domain']].append(role_type)

    def frame_lemma_direct(self, frame, lemma):
        return aida_maps.frame_lemma_map.get((frame, lemma), None)

    def head_token_direct(self, lemma):
        if lemma in aida_maps.token_direct_map:
            return aida_maps.token_direct_map[lemma]

    def arg_direct(self, content, entities):
        for arg in content['arguments']:
            arg_lemma = entities[arg['entityId']]['lemma']
            if arg_lemma in aida_maps.arg_direct_map:
                return aida_maps.arg_direct_map[arg_lemma][0]

    def map_from_event_type(self, event_type, lemma):
        # print("mapping tac kbp event type ", event_type, lemma)

        level1, level2 = event_type.split('_')
        level2_tokens = event_type_split(level2)

        l_score, m_score, full_type = self.map_by_pred_match(
            [t + '-pred' for t in level2_tokens], [lemma + '-pred'])

        if m_score > 0.8 or l_score > 0.8:
            if l_score > 0.8:
                return full_type
            else:
                if full_type in self.type_parent:
                    # print('parent is ', self.type_parent[full_type])
                    return self.type_parent[full_type]
                else:
                    # print('no parent')
                    return full_type

    def map_from_lemma_only(self, lemma):
        l_score, m_score, full_type = self.map_by_pred_match(
            [lemma + '-pred'], [lemma + '-pred'])

        if m_score > 0.8 or l_score > 0.8:
            if l_score > 0.8:
                return full_type
            else:
                if full_type in self.type_parent:
                    return self.type_parent[full_type]
                else:
                    return full_type

    def map_from_frame(self, frame, lemma):
        if frame in aida_maps.frame_direct_map:
            return aida_maps.frame_direct_map[frame]

        lemma_pred = lemma + '_pred'
        # print(frame, lemma_pred)
        l_score, m_score, full_type = self.map_by_pred_match(
            [frame], [frame, lemma_pred]
        )

        if m_score > 0.7 or l_score > 0.7:
            if l_score > 0.7:
                return full_type
            else:
                if full_type in self.type_parent:
                    return self.type_parent[full_type]
                else:
                    return full_type

    def map_by_pred_match(self, middle_matchers, low_matchers):
        scored_pairs = []

        for onto_type, onto_type_tokens in self.onto_event_tokens.items():
            middle_score = 0
            low_score = 0

            mid_tokens = onto_type_tokens.get('middle', []) + \
                         onto_type_tokens.get('top', [])

            for onto_t in mid_tokens:
                if onto_t in self.pred_embeds:
                    onto_emd = self.pred_embeds[onto_t]
                    for t in middle_matchers:
                        t_id = self.resources.event_embed_vocab.get_index(
                            t, None)
                        if t_id >= 0:
                            t_emd = self.resources.event_embedding[t_id]
                            s = 1 - cosine(onto_emd, t_emd)

                            if s > middle_score:
                                middle_score = s

            low_tokens = onto_type_tokens.get('low', [])
            for onto_t in low_tokens:
                if onto_t in self.pred_embeds:
                    onto_emd = self.pred_embeds[onto_t]
                    for lemma_pred in low_matchers:
                        lemma_idx = self.resources.event_embed_vocab.get_index(
                            lemma_pred, None
                        )

                        if lemma_idx >= 0:
                            emd = self.resources.event_embedding[lemma_idx]
                            s = 1 - cosine(onto_emd, emd)
                            if s > low_score:
                                low_score = s

            rank_num = max(low_score, middle_score)

            scored_pairs.append((rank_num, low_score, middle_score, onto_type))

        scored_pairs.sort(reverse=True)

        for rank_num, low_score, mid_score, t in scored_pairs:
            if mid_score < 0.2:
                continue
            return low_score, mid_score, t

        return 0, 0, None

    def map_event_type(self, event, entities):
        event_head = event['headLemma']

        head_direct_type = self.head_token_direct(event_head)
        if head_direct_type:
            return 'head_direct', head_direct_type

        if event['component'] == 'CrfMentionTypeAnnotator':
            t = event['type']
            if (t, event.get('frame', 'NA')) in aida_maps.kbp_frame_correction:
                t = aida_maps.kbp_frame_correction[t, event['frame']]

            if (t, event_head) in aida_maps.kbp_lemma_map:
                t = aida_maps.kbp_lemma_map[(t, event_head)]
                return 'map_kbp_lemma', t

            if t in aida_maps.kbp_direct_map:
                return 'map_kbp_direct', aida_maps.kbp_direct_map[t]

        arg_direct_type = self.arg_direct(event, entities)
        if arg_direct_type:
            return 'arg_direct', arg_direct_type

        if 'frame' in event:
            t = self.frame_lemma_direct(event['frame'], event['headLemma'])
            if t:
                return 'map_from_frame', t

            t = self.map_from_frame(event['frame'], event['headLemma'])
            if t:
                return 'map_from_frame', t

        if event['component'] == 'CrfMentionTypeAnnotator':
            # Mapping from event map is less reliable.
            t = event['type']
            t = self.map_from_event_type(t, event['headLemma'])
            if t:
                return 'map_from_event_type', t

            if t in aida_maps.kbp_backup_map:
                return 'map_kbp_backup', aida_maps.kbp_backup_map[t]

        if event['component'] == 'VerbBasedEventDetector':
            t = self.map_from_lemma_only(event['headLemma'])
            if t:
                return 'map_from_head_lemma', t

    def map_arg_role(self, evm, arg, entities):
        arg_lemma = entities[arg['entityId']]['lemma']
        event_type = evm['type']
        event_head = evm['headLemma']

        # List the event types in a hierarchy, with the specific one first. In
        # such cases, we will look for the low ontology items first.
        l_types = [event_type]
        t_parts = event_type.split('.')
        if len(t_parts) > 2:
            l_types.append('.'.join(t_parts[:2]))
        if len(t_parts) > 1:
            l_types.append('.'.join(t_parts[:1]))
        l_types.append(t_parts[0])

        # List the roles, with frame element first.
        # We trust fn more, but "other" the least, so they are at the end.
        l_roles = []

        other_roles = []
        for role in arg['roles']:
            prefix, r = role.split(':', 1)
            if prefix == 'fn':
                l_roles.insert(0, r)
            elif prefix == 'other':
                other_roles.append(r)
            else:
                l_roles.append(r)

        l_roles.extend(other_roles)

        if arg_lemma in aida_maps.arg_direct_map:
            if event_type == aida_maps.arg_direct_map[arg_lemma][0]:
                return aida_maps.arg_direct_map[arg_lemma][1]
        else:
            for role in l_roles:
                # Go through the event type hierarchy, then go up.
                for t in l_types:
                    if role in aida_maps.srl_ldc_arg_map.get(t, {}):
                        return event_type + '_' + \
                               aida_maps.srl_ldc_arg_map[t][role]
                    if role == 'ARGM-LOC' or role == 'Place':
                        return f'{event_type}_Place'
                else:
                    if role == 'ARGM-TMP' or role == 'Time':
                        continue
                    debug_file.write(
                        f"frame: {evm.get('frame', 'no_frame')} , "
                        f"head: {event_head} , "
                        f"arg: {arg_lemma} , "
                        f"role: {role} , "
                        f"event type: {event_type}\n"
                    )


def process_one(type_mapper, resources, fin, fout):
    rich_doc = json.load(fin)
    # text = rich_doc['text']

    entities = {}
    for ent in rich_doc['entityMentions']:
        entities[ent['id']] = {
            'span': ent['span'],
            "head_span": ent['headWord']['span'],
            "text": ent['text'],
            "lemma": ent['headWord']['lemma'],
            'id': ent['id'],
        }

    for evm in rich_doc['eventMentions']:
        # print('evm is **' + evm['headLemma'] + '**')
        map_res = type_mapper.map_event_type(evm, entities)
        if not map_res:
            # print('No mapping results')
            continue

        # print('Mapping results is ', map_res)
        # input('take a look')

        rule, mapped_type = map_res

        if mapped_type and mapped_type in resources.onto_set:
            evm['type'] = mapped_type

            for arg in evm['arguments']:
                roles = []
                mapped_role = type_mapper.map_arg_role(
                    evm, arg, entities)

                if mapped_role:
                    if mapped_role in resources.onto_set:
                        roles.append(mapped_role)
                    else:
                        debug_file.write(f'Mapped role not valid: '
                                         f'{mapped_role}\n')

                roles.extend(arg['roles'])
                arg['roles'] = roles

    json.dump(rich_doc, fout)


def main(para, resources):
    type_mapper = ZeroShotTypeMapper(resources)

    if not os.path.exists(para.output_path):
        os.makedirs(para.output_path)

    for p in os.listdir(para.input_path):
        if not p.endswith('.json'):
            continue
        with open(os.path.join(para.input_path, p)) as fin, \
                open(os.path.join(para.output_path, p), 'w') as fout:
            try:
                process_one(type_mapper, resources, fin, fout)
            except Exception as err:
                sys.stderr.write(
                    f"ERROR: Exception in ZeroShotPredictor while "
                    f"processing p\n")
                traceback.print_exc()
                logging.error(traceback.format_exc())


if __name__ == '__main__':
    class Basic(Configurable):
        input_path = Unicode(help='Input path.').tag(config=True)
        output_path = Unicode(help='Output path.').tag(config=True)


    debug_file = tempfile.NamedTemporaryFile(suffix='_zero_shot_event_debug.txt', mode='w', delete=False)

    set_basic_log()
    conf = load_mixed_configs()
    basic_para = Basic(config=conf)
    res = ZeroShotEventResources(config=conf)

    main(basic_para, res)

    debug_file.close()
