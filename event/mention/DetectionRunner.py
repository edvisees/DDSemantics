from event.io.readers import (
    ConllUReader,
    Vocab
)
from event.util import set_basic_log
from event.mention.models.detectors import (
    TextCNN,
    FrameMappingDetector
)
import logging
import torch
import torch.nn.functional as F
import os
import json


class DetectionRunner:
    def __init__(self, config, token_vocab, tag_vocab):
        self.model_dir = config.model_dir
        self.model_name = config.model_name
        self.trainable = True

        self.init_model(config, token_vocab, tag_vocab)

    def init_model(self, config, token_vocab, tag_vocab):
        if self.model_name == 'cnn':
            self.model = TextCNN(config, tag_vocab.vocab_size(),
                                 token_vocab.vocab_size())
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.model_name == 'frame':
            self.model = FrameMappingDetector(config, token_vocab)
            self.trainable = False

    def train(self, train_reader, dev_reader):
        if not self.trainable:
            return

        optimizer = torch.optim.Adam(self.model.parameters())
        self.model.train()

        epoch = 10
        step = 0
        best_step = 0
        log_every_k = 10

        early_patience = 10

        best_res = 0

        for epoch in range(epoch):
            input, labels = train_reader.read_batch()
            optimizer.zero_grad()

            # print("Batch:")
            # print(input)
            # print(labels)
            logits = self.model(input)
            loss = F.cross_entropy(logits, labels)
            loss.backward()

            optimizer.step()
            step += 1

            # Eval on dev.
            if not step % log_every_k:
                dev_res = self.eval(dev_reader)

                if dev_res > best_res:
                    best_res = dev_res
                    best_step = step
                    self.save()
                else:
                    if step - best_step > early_patience:
                        logging.info(
                            "Early stop with patience %d." % early_patience
                        )

        train_reader.tag_vocab.fix()

    def eval(self, dev_reader):
        return 0

    def save(self):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, self.model_name)
        torch.save(self.model, path)

    def predict(self, test_reader, collector):
        event_id = 0
        entity_id = 0
        for data in test_reader.read_window():
            tokens, tags, features = data
            # print(test_reader.reveal_origin(tokens))
            # print(test_reader.reveal_tags(tags))
            l_tags, l_args = self.model.predict(data)

            center = int(len(tokens) / 2)

            token_id = tokens[center]
            feature = features[center]
            span = feature[-1]
            event_lemma = feature[0]

            for t, args in zip(l_tags, l_args):
                if not t == "O":
                    role_ids = {}
                    for role, (entity_lemma, span, entity_type) in args.items():
                        entity_id += 1
                        entity_details = {
                            'entity_type': entity_type,
                            'surface': entity_lemma
                        }
                        eid = "Entity_%d" % entity_id

                        collector.add_interp(eid, 'Entity', span,
                                             entity_details)
                        role_ids[role] = eid

                    event_details = {
                        'arguments': {},
                        'surface': event_lemma,
                        'event_type': t
                    }
                    for role, role_id in role_ids.items():
                        event_details['arguments'][role] = role_id

                    collector.add_interp("Event_%d" % event_id, 'Event', span,
                                         event_details)

                    event_id += 1


class InterpCollector:
    def __init__(self, docid, out_path):
        self.info = {
            "id": docid,
            "interp": []
        }
        self.out_path = out_path

    def add_interp(self, interp_id, record_type, char_span, details):
        interp = {
            'id': interp_id,
            'record_type': record_type,
            'char_span': char_span,
        }
        for k, v in details.items():
            interp[k] = v

        self.info['interp'].append(interp)

    def write(self):
        with open(self.out_path, 'w') as out:
            json.dump(self.info, out, indent=2)


def main(config):
    token_vocab = Vocab(config.experiment_folder, 'tokens',
                        embedding_path=config.word_embedding,
                        emb_dim=config.word_embedding_dim)

    tag_vocab = Vocab(config.experiment_folder, 'tag',
                      embedding_path=config.tag_list)

    train_reader = ConllUReader(config.train_files, config, token_vocab,
                                tag_vocab)
    dev_reader = ConllUReader(config.dev_files, config, token_vocab,
                              train_reader.tag_vocab)
    detector = DetectionRunner(config, token_vocab, tag_vocab)
    detector.train(train_reader, dev_reader)

    res_collector = InterpCollector('bellingcat', config.output)

    test_reader = ConllUReader(config.test_files, config, token_vocab,
                               train_reader.tag_vocab)
    detector.predict(test_reader, res_collector)

    res_collector.write()


if __name__ == '__main__':
    from event.util import OptionPerLineParser

    parser = OptionPerLineParser(description='Event Mention Detector.',
                                 fromfile_prefix_chars='@')

    parser.add_argument('--model_name', type=str)

    parser.add_argument('--experiment_folder', type=str)
    parser.add_argument('--model_dir', type=str)

    parser.add_argument('--train_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--dev_files',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--test_files',
                        type=lambda s: [item for item in s.split(',')])

    parser.add_argument('--output', type=str)

    parser.add_argument('--word_embedding', type=str,
                        help='Word embedding path')
    parser.add_argument('--word_embedding_dim', type=int, default=300)

    parser.add_argument('--position_embedding_dim', type=int, default=50)

    parser.add_argument('--tag_list', type=str,
                        help='Frame embedding path')
    parser.add_argument('--tag_embedding_dim', type=int, default=50)

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout [default: 0.5]')
    parser.add_argument('--context_size', default=30, type=int)
    parser.add_argument('--window_sizes', default='2,3,4,5',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--filter_num', default=100, type=int,
                        help='Number of filters for each type.')
    parser.add_argument('--fix_embedding', type=bool, default=False)

    parser.add_argument('--batch_size', type=int, default=50)

    parser.add_argument('--format', type=str, default="conllu")
    parser.add_argument('--no_punct', type=bool, default=False)
    parser.add_argument('--no_sentence', type=bool, default=False)

    # Frame based detector.
    parser.add_argument('--frame_lexicon', type=str, help='Frame lexicon path')
    parser.add_argument('--event_list', help='Lexicon for events', type=str)
    parser.add_argument('--entity_list', help='Lexicon for entities', type=str)
    parser.add_argument('--relation_list', help='Lexicon for relations',
                        type=str)

    arguments = parser.parse_args()

    set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
