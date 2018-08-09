import math
from event.io.ontology import OntologyLoader


class MentionDetector:
    def __init__(self, **kwargs):
        super().__init__()
        self.unknown_type = "UNKNOWN"

    def predict(self, *input):
        pass


class BaseRuleDetector(MentionDetector):
    def __init__(self, config, token_vocab):
        super().__init__()
        self.onto = OntologyLoader(config.ontology_path)
        self.token_vocab = token_vocab

    def predict(self, *input):
        words, _, l_feature, word_meta, sent_meta = input[0]
        center = math.floor(len(words) / 2)
        lemmas = [features[0] if features else '' for features in l_feature]
        words = self.token_vocab.reveal_origin(words)
        return self.predict_by_word(words, lemmas, l_feature, center)

    def predict_by_word(self, words, lemmas, l_feature, center):
        raise NotImplementedError('Please implement the rule detector.')


class MarkedDetector(BaseRuleDetector):
    """
    Assume there is one field already marked with event type.
    """

    def __init__(self, config, token_vocab, marked_field_index=-2):
        super().__init__(config, token_vocab)
        self.marked_field_index = marked_field_index

    def predict(self, *input):
        words, _, l_feature, word_meta, sent_meta = input[0]
        center = math.floor(len(words) / 2)
        return l_feature[center][self.marked_field_index]


class FrameMappingDetector(BaseRuleDetector):
    def __init__(self, config, token_vocab):
        super().__init__(config, token_vocab)
        self.lex_mapping = self.load_frame_lex(config.frame_lexicon)
        self.entities, self.events, self.relations = self.load_wordlist(
            config.entity_list, config.event_list, config.relation_list
        )

        self.temp_memory = set()

    def load_frame_lex(self, frame_path):
        import xml.etree.ElementTree as ET
        import os

        ns = {'berkeley': 'http://framenet.icsi.berkeley.edu'}

        lex_mapping = {}

        for file in os.listdir(frame_path):
            with open(os.path.join(frame_path, file)) as f:
                tree = ET.parse(f)
                frame = tree.getroot()
                frame_name = frame.get('name')

                lexemes = []
                for lexUnit in frame.findall('berkeley:lexUnit', ns):
                    lex = lexUnit.get('name')
                    lexeme = lexUnit.findall(
                        'berkeley:lexeme', ns)[0].get('name')
                    lexemes.append(lexeme)

                lex_text = ' '.join(lexemes)
                if lex_text not in lex_mapping:
                    lex_mapping[lex_text] = []

                lex_mapping[lex_text].append(frame_name)
        return lex_mapping

    def load_wordlist(self, entity_file, event_file, relation_file):
        events = {}
        entities = {}
        relations = {}
        with open(event_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, ontology = line.strip().split()
                    events[word] = ontology
        with open(entity_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) == 2:
                    word, ontology = line.strip().split()
                    entities[word] = ontology
        with open(relation_file) as fin:
            for line in fin:
                parts = line.strip().split()
                if parts:
                    event_type = parts[0]
                    args = parts[1:]

                    if event_type not in relations:
                        relations[event_type] = {}

                    for arg in args:
                        arg_role, arg_types = arg.split(":")
                        relations[event_type][arg_role] = arg_types.split(",")

        return entities, events, relations

    def predict_by_word(self, words, lemmas, l_feature, center):
        event_types = []

        center_word = words[center]
        center_lemma = lemmas[center]

        # if center_lemma in self.lex_mapping:
        #     for fname in self.lex_mapping[center_lemma]:
        #         event_types.append(('FrameNet', fname))

        # if center_word in self.events:
        #     event_type = self.events[center_word]
        #
        # if center_lemma in self.events:
        #     event_type = self.events[center_lemma]
        # pos_list = [features[1] for features in l_feature]
        # deps = [(features[2], features[3]) for features in l_feature]
        # if not event_type == self.unknown_type:
        #     res = self.predict_args(center, event_type, lemmas, pos_list,
        #                             deps)
        #
        #     for role, entity in res.items():
        #         if entity:
        #             index, entity_type = entity
        #             features = l_feature[index]
        #             args[role] = index, entity_type

        return event_types

    def predict_args(self, center, event_type, context, pos_list, deps):
        if event_type not in self.relations:
            return {}

        expected_relations = self.relations[event_type]
        expected_relations["Location"] = ["Loc", "GPE"]
        expected_relations["Time"] = ["Time"]

        filled_relations = dict([(k, None) for k in expected_relations])
        num_to_fill = len(filled_relations)

        relation_lookup = {}
        for role, types in expected_relations.items():
            for t in types:
                relation_lookup[t] = role

        for distance in range(1, center + 1):
            left = center - distance
            right = center + distance

            left_lemma = context[left]
            right_lemma = context[right]

            if left_lemma in self.entities:
                arg_type = self.check_arg(context[center], event_type,
                                          left_lemma, deps)
                if arg_type in relation_lookup:
                    possible_rel = relation_lookup[arg_type]
                    if filled_relations[possible_rel] is None:
                        filled_relations[possible_rel] = (left, arg_type)
                        num_to_fill -= 1

            if right_lemma in self.entities:
                arg_type = self.check_arg(context[center], event_type,
                                          right_lemma, deps)
                if arg_type in relation_lookup:
                    possible_rel = relation_lookup[arg_type]
                    if filled_relations[possible_rel] is None:
                        filled_relations[possible_rel] = (right, arg_type)
                        num_to_fill -= 1

            if num_to_fill == 0:
                break

        return filled_relations

    def check_arg(self, predicate, event_type, arg_lemma, features):
        unknown_type = "O"

        entity_type = unknown_type
        if arg_lemma in self.entities:
            entity_type = self.entities[arg_lemma]

        if not entity_type == unknown_type:
            return entity_type

        return None
