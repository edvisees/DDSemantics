import logging
import math
import pdb

from torch import nn
from torch.nn import functional as F
import torch
from texar.torch.modules.encoders import TransformerEncoder
from texar.torch.modules.embedders import EmbedderBase

from event.nn.models import KernelPooling
from event import torch_util
from conf.implicit import texar_config
from event.arguments.implicit_arg_params import ArgModelPara

logger = logging.getLogger(__name__)


class ArgCompatibleModel(nn.Module):
    """ """

    def __init__(self, para, resources, device, model_name):
        super(ArgCompatibleModel, self).__init__()
        self.para = para

        # This will be used to create the one-hot vector for the slots.
        self.num_slots = para.num_slots

        self.event_embedding = None
        self.word_embedding = None

        self.device = device

        self.name = model_name

        self.__load_embeddings(resources)

    def self_event_mask(self, batch_event_indices, batch_size, event_size,
                        context_size):
        """Return a matrix to mask out the scores from the current event.

        Args:
          batch_event_indices: batch_size
          event_size: context_size
          batch_size: 
          context_size: 

        Returns:
          

        """
        selector = batch_event_indices.unsqueeze(-1)
        one_zeros = torch.ones(
            batch_size, event_size, context_size, dtype=torch.float32,
        ).to(self.device)
        one_zeros.scatter_(-1, selector, 0)
        return one_zeros

    def __load_embeddings(self, resources):
        logger.info("Loading %d x %d event embedding." % (
            resources.event_embed_vocab.get_size(),
            self.para.event_embedding_dim
        ))

        # Add additional dimension for extra event vocab.
        self.event_embedding = nn.Embedding(
            resources.event_embed_vocab.get_size(),
            self.para.event_embedding_dim,
        )

        logger.info("Loading %d x %d word embedding." % (
            resources.word_embed_vocab.get_size(),
            self.para.word_embedding_dim
        ))

        self.word_embedding = nn.Embedding(
            resources.word_embed_vocab.get_size(),
            self.para.word_embedding_dim,
            padding_idx=self.para.word_vocab_size
        )

        if resources.word_embedding_path is not None:
            word_embed = torch.from_numpy(resources.word_embedding)

            # Add extra dimensions for word padding.
            zeros = torch.zeros(resources.word_embed_vocab.extra_size(),
                                self.para.word_embedding_dim)
            self.word_embedding.weight = nn.Parameter(
                torch.cat((word_embed, zeros))
            )

        if resources.event_embedding_path is not None:
            event_emb = torch.from_numpy(resources.event_embedding)

            # Add extra event vocab for unobserved args.
            zeros = torch.zeros(resources.event_embed_vocab.extra_size(),
                                self.para.event_embedding_dim)
            self.event_embedding.weight = nn.Parameter(
                torch.cat((event_emb, zeros))
            )


class RandomBaseline(ArgCompatibleModel):
    """ """

    def __init__(self, para, resources, device):
        super(RandomBaseline, self).__init__(para, resources, device,
                                             'random_baseline')

    def forward(self, batch_event_data, batch_info):
        """

        Args:
          batch_event_data: 
          batch_info: 

        Returns:

        """
        batch_features = batch_event_data['features']
        a, b, c = batch_features.shape
        return torch.rand(a, b, 1)


class MostFrequentModel(ArgCompatibleModel):
    """ """

    def __init__(self, para, resources, device):
        super(MostFrequentModel, self).__init__(para, resources, device,
                                                'most_freq_baseline')
        self.para = para

    def forward(self, batch_event_data, batch_info):
        """

        Args:
          batch_event_data: 
          batch_info: 

        Returns:

        """
        # batch x instance_size x n_features
        batch_features = batch_event_data['features']
        return batch_features[:, :, 7]


class BaselineEmbeddingModel(ArgCompatibleModel):
    """ """

    def __init__(self, para, resources, device):
        super(BaselineEmbeddingModel, self).__init__(para, resources, device,
                                                     'w2v_baseline')
        self.para = para

        self._score_method = para.w2v_baseline_method
        self._avg_topk = para.w2v_baseline_avg_topk

    def forward(self, batch_event_data, batch_info):
        """

        Args:
          batch_event_data: 
          batch_info: 

        Returns:

        """
        # batch x instance_size x event_component
        batch_event_rep = batch_event_data['event_components']

        # batch x context_size x event_component
        batch_context = batch_info['context_events']

        batch_event_indices = batch_info['event_indices']

        # Add embedding dimension at the end.
        # batch x context_size x event_component x embedding
        context_emb = self.event_embedding(batch_context)

        # batch x instance_size x event_component x embedding
        event_emb = self.event_embedding(batch_event_rep)

        # This is the concat way:
        # batch x context_size x embedding_x_component
        if self.para.w2v_event_repr == 'concat':
            flat_context_emb = context_emb.view(
                context_emb.size()[0], -1,
                context_emb.size()[-1] * context_emb.size()[-2]
            )

            # batch x instance_size x embedding_x_component
            flat_event_emb = event_emb.view(
                event_emb.size()[0], -1,
                event_emb.size()[-1] * event_emb.size()[-2]
            )
        elif self.para.w2v_event_repr == 'sum':
            # Sum all the components together.
            # batch x instance_size x embedding
            flat_context_emb = context_emb.sum(-2)
            flat_event_emb = event_emb.sum(-2)
        else:
            raise ValueError(f"Unknown event representation method: "
                             f"[{self.para.w2v_event_repr}]")

        # Compute cosine.
        nom_event_emb = F.normalize(flat_event_emb, 2, -1)
        nom_context_emb = F.normalize(flat_context_emb, 2, -1)

        bs, event_size, _ = batch_event_rep.shape
        bs, context_size, _ = batch_context.shape
        self_mask = self.self_event_mask(batch_event_indices, bs, event_size,
                                         context_size)

        # Cosine similarities to the context.
        # batch x instance_size x context_size
        trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        trans *= self_mask

        # batch x instance_size
        if self._score_method == 'max_sim':
            pooled, _ = trans.max(2, keepdim=False)
        elif self._score_method == 'average':
            pooled, _ = trans.mean(2, keepdim=False)
        elif self._score_method == 'topk_average':
            topk_pooled = torch_util.topk_with_fill(
                trans, self.para.w2v_baseline_avg_topk, 2, largest=True)
            pooled = topk_pooled.mean(2, keepdim=False)
        else:
            raise ValueError("Unknown method.")

        return pooled


class ArgPositionEmbedder(EmbedderBase):
    """Embedder for argument slots. This can be an embeder for the frame slots, or
    an embeder for theargument positions as well.

    Args:

    Returns:

    """

    def __init__(self, embeddings, hparams=None):
        super().__init__(hparams=None)
        # The embedding dimension.
        self._dim = hparams.dim
        self._embeddings = embeddings

    @property
    def output_size(self) -> int:
        """ """
        return self._dim

    def forward(self, role_indices: torch.LongTensor) -> torch.Tensor:
        """Embed a list of slot role indices into the role embeddings.

        Args:
          role_indices: Input is the a tensor of the role ids, in the
        shape of batch x sequence_length
          role_indices: torch.LongTensor:
          role_indices: torch.LongTensor:
          role_indices: torch.LongTensor: 

        Returns:
          : A embedded tensor of shape batch x sequence_length x embedding_dim

        """
        return self._embeddings(role_indices)


class RoleArgCombineModule(nn.Module):
    """
    Combine the role and arguments embeddings.

    Args:
        role_combine_method:
        embedding_dim:
    """

    def __init__(self, role_combine_method, embedding_dim):
        super().__init__()
        self._role_combine_method = role_combine_method

        self.weighting_layers = None
        if self._role_combine_method == 'mlp':
            self.weighting_layers = MLP(embedding_dim * 2, [embedding_dim])

    def forward(self, role_repr, arg_repr):
        """

        Args:
          role_repr: 
          arg_repr: 

        Returns:

        """
        if self._role_combine_method == 'add':
            res = role_repr + arg_repr
        elif self._role_combine_method == 'cat':
            res = torch.cat((role_repr, arg_repr), -1)
        elif self._role_combine_method == 'mlp':
            res = self.weighting_layers(role_repr, arg_repr)
        elif self._role_combine_method == 'biaffine':
            raise NotImplementedError("I'm unclear how biaffine work here.")
        else:
            raise NotImplementedError(
                f"Unsupported type {self._role_combine_method}")

        return res


class DynamicEventReprModule(nn.Module):
    """ """

    def __init__(self, para: ArgModelPara):
        super().__init__()

        # Use Texar Transformer.
        # Texar module are subclass of Pytorch Modules.
        self._transformer = TransformerEncoder(
            hparams=texar_config.arg_transformer,
        )
        self._role_arg_combiner = RoleArgCombineModule(
            para.arg_role_combine_func, para.event_embedding_dim)

        self._pred_arg_mlp = MLP(para.event_embedding_dim * 3,
                                 para.arg_composition_layer_sizes)
        self.output_dim = para.arg_composition_layer_sizes[-1]

    def multi_slot_combine_func(self, arg_repr, arg_mask):
        """
        Combine the variable length arguments into one fixed length vector.

        Args:
            arg_repr: The argument representations.
            arg_mask: The argument mask.

        Returns:

        """
        # A sum based combine function. This assumes that the padded value in
        # arg_repr are all zero.
        # TODO: make sure assumption is true.
        return torch.sum(arg_repr, dim=2) / arg_mask.unsqueeze(-1).float()

    def forward(self, event_data):
        """Compute the transformer encoded output of roles and the args.

        Args:
          event_data: A dict to a series of tensors representing the predicate
          and argument.

            - predicate: A tensor of the predicate embedding of shape
              batch x instance x embedding_dim
            - frame: A tensor of the frame embedding of shape
              batch x instance x embedding_dim
            - slot: A tensor of the role representations of shape
              batch x instance x num_roles (padded) x embedding_dim
            - slot_value: A tensor of the arg representations of shape
              batch x instance x num_roles (padded) x embedding_dim
            - slot_length: A tensor containing the actual length of each

        Returns:
          : The combined and pooled result of the argument and role pairs,
          : The combined and pooled result of the argument and role pairs,
          of shape batch x embedding_dim

        """
        # This combines the slot name and slot value, which mimics the
        # positional encoding idea. The output shape is
        # batch x #instance x #slots x embedding_dim.
        combined_arg_role = self._role_arg_combiner(event_data['slot'],
                                                    event_data['slot_value'])

        # We have the argument representation now, which is weighted by the
        # slot names, now we pass them into the transformer.
        b, i, s, e = combined_arg_role.shape

        # Before passing to the transformer, we view the batch and instance
        # dimension as the batch dimention only.
        self_att_args = self._transformer(
            combined_arg_role.view(b * i, s, e),
            event_data['slot_length'].view(b * i, 1)
        ).view(b, i, s, e)

        combined_args = self.multi_slot_combine_func(
            self_att_args, event_data['slot_length'])

        return self._pred_arg_mlp(event_data['predicate'], event_data['frame'],
                                  combined_args)


class FixedEventReprModule(nn.Module):
    """ """

    def __init__(self, para: ArgModelPara):
        super().__init__()
        component_per = 2 if para.use_frame else 1
        num_event_components = (1 + para.num_slots) * component_per
        self.arg_comp = MLP(
            para.event_embedding_dim * num_event_components,
            para.arg_composition_layer_sizes
        )
        self.output_dim = para.arg_composition_layer_sizes[-1]

    def forward(self, event_data):
        """

        Args:
          event_data:

        Returns:

        """
        # In fixed slot mode, the argument composition is simply done via
        # a MLP since the dimension is fixed.
        flatten_embedding_size = event_data.size()[-1
                                 ] * event_data.size()[-2]
        flatten_pred_emb = event_data.view(
            event_data.size()[0], -1, flatten_embedding_size)
        return self.arg_comp(flatten_pred_emb)


class MLP(nn.Module):
    """ """

    def __init__(self, input_hidden_size, output_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        input_size = input_hidden_size
        for output_size in output_sizes:
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

    def forward(self, *input_data, activation=F.relu):
        """

        Args:
          *input_data: 
          activation: (Default value = F.relu)

        Returns:

        """
        _data = torch.cat(input_data, -1)

        for layer in self.layers:
            _data = activation(layer(_data))
        return _data


class Biaffine(nn.Module):
    """
    Biaffine layer (actually only bilinear now).

    Args:
        size_l:
        size_b:
    """

    def __init__(self, size_l, size_b):
        super(Biaffine, self).__init__()

        self._m_affine = nn.Linear(size_l, size_b)

    def forward(self, tensor_l: torch.Tensor, tensor_r: torch.Tensor):
        """

        Args:
          tensor_l: torch.Tensor:
          tensor_r: torch.Tensor:

        Returns:

        """
        return torch.bmm(self._m_affine(tensor_l), tensor_r.transpose(-2, -1))


class PredicateWindowModule(nn.Module):
    """ """

    def __init__(self, para: ArgModelPara):
        super().__init__()
        self.event_to_var_layer = nn.Linear(
            self.para.event_embedding_dim, para.num_distance_features
        )
        # Each distance measure will be rescaled into a probability score.
        self.output_dim = para.num_distance_features

        self.sqrt_2pie = math.sqrt(2 * math.pi)

    def forward(self, predicates, distances):
        """Compute the probabilities of the instance given the distance and
        predicate.

        Args:
          predicates: The tensor containing the predicates embeddings,
        of shape batch x instance_size x embedding_dim
          distances: The tensor containing the distance features,
        of shape batch x instance_size x #distance_measures

        Returns:

        """
        # The intuition here is that the distance is dependent on the predicate,
        # we assume that the probability factor follows a certain
        # distribution regarding the distance to the current sentence, and the
        # predicate determine the variance of the distribution.
        # We do not intend to update the predicate representation here so we
        # detach it.
        variances = self.event_to_var_layer(predicates.detach())

        dist_sq = distances * distances

        # The gaussian based of the distance function.
        return torch.exp(- 0.5 * dist_sq / variances) / (
                self.sqrt_2pie * variances)


class EventContextAttentionPool(nn.Module):
    """ """

    def __init__(self, para: ArgModelPara):
        super().__init__()

        # Config feature size.
        self._vote_pool_type = para.vote_pooling
        if self._vote_pool_type == 'kernel':
            self._kp = KernelPooling()
            # The output dimensions are the K kernel values.
            self.output_dim = self._kp.K
        elif self._vote_pool_type == 'topk':
            self._pool_topk = para.pool_topk
            # The output dimensions are the top k values.
            self.output_dim = para.pool_topk
        else:
            # The output dim is 1.
            self.output_dim = 1

        self._vote_method = para.vote_method

        if self._vote_method == 'biaffine':
            self.event_vote_layer = Biaffine(self.para.event_embedding_dim,
                                             self.para.event_embedding_dim)
        elif self._vote_method == 'mlp':
            raise NotImplementedError("MLP not yet supported when voting.")

    def _context_vote(self, nom_event_emb, nom_context_emb):
        """

        Args:
          nom_event_emb: 
          nom_context_emb: 

        Returns:

        """
        # First compute the trans matrix between events and the context.
        if self._vote_method == 'cosine':
            # Normalized dot product is cosine.
            trans = torch.bmm(nom_event_emb, nom_context_emb.transpose(-2, -1))
        elif self._vote_method == 'biaffine':
            trans = self.event_vote_layer(nom_event_emb, nom_context_emb)
        elif self._vote_method == 'mlp':
            raise NotImplementedError("MLP not yet supported when voting.")
        else:
            raise ValueError(
                'Unknown vote computation method {}'.format(self._vote_method)
            )
        return trans

    def forward(self, event_emb, context_emb, self_avoid_mask):
        """Compute the contextual scores in the attentive way, i.e., computing
        some cross scores between the two representations.

        Args:
          event_emb: context_emb:
          self_avoid_mask: mask of shape event_size x context_size, each
        row is contain only one zero that indicate which context should not be
        used.
          context_emb: 

        Returns:

        """
        nom_event_emb = F.normalize(event_emb, 2, -1)
        nom_context_emb = F.normalize(context_emb, 2, -1)

        # TODO: half of both matrixes are zero, probably something wrong.
        trans = self._context_vote(nom_event_emb, nom_context_emb)

        pdb.set_trace()

        # Make the self score zero.
        # TODO: self_avoid didn't work here.
        trans = trans * self_avoid_mask

        if self._vote_pool_type == 'kernel':
            # TODO: kp gives nan.
            pooled_value = self._kp(trans)
        elif self._vote_pool_type == 'max':
            pooled_value, _ = trans.max(2, keepdim=True)
        elif self._vote_pool_type == 'average':
            pooled_value = trans.mean(2, keepdim=True)
        elif self._vote_pool_type == 'topk':
            if trans.shape[2] >= self._pool_topk:
                pooled_value, _ = trans.topk(self._pool_topk, 2,
                                             largest=True)
            else:
                added = torch.zeros(
                    (trans.shape[0], trans.shape[1],
                     self._pool_topk - trans.shape[2])).to(self.device)
                pooled_value = torch.cat((trans, added), -1)
        else:
            raise ValueError(
                'Unknown pool type {}'.format(self._vote_pool_type)
            )
        return pooled_value


class EventCoherenceModel(ArgCompatibleModel):
    """ """

    def __init__(self, para: ArgModelPara, resources, device, model_name):
        super(EventCoherenceModel, self).__init__(para, resources, device,
                                                  model_name)
        logger.info(f"Pair composition network {model_name} started, "
                    f"with {self.para.num_extracted_features} extracted"
                    f" features and {self.para.num_distance_features} "
                    f"distance features.")

        # Number of extracted discrete features.
        feature_size = self.para.num_extracted_features

        if self.para.arg_representation_method == 'fix_slots':
            self.arg_composition_model = FixedEventReprModule(para)
        elif self.para.arg_representation_method == 'role_dynamic':
            self.arg_composition_model = DynamicEventReprModule(para)
        else:
            raise ValueError(f"Unknown arg representation method"
                             f" {self.arg_representation_method}")

        self.context_vote_layer = EventContextAttentionPool(para)

        feature_size += self.arg_composition_model.output_dim
        feature_size += self.context_vote_layer.output_dim

        # Dim for 1-hot slot position feature.
        if self.para.arg_representation_method == 'fix_slots':
            feature_size += self.num_slots
        else:
            # Dim for the FE representation.
            feature_size += self.para.event_embedding_dim

        self._use_distance = para.encode_distance
        if self._use_distance:
            self.distance_module = PredicateWindowModule(para)
            feature_size += self.distance_module.output_dim

        self._linear_combine = nn.Linear(feature_size, 1)

        if para.loss == 'cross_entropy':
            self.normalize_score = True
        else:
            self.normalize_score = False

    def event_repr(self, batch_event_data, batch_info):
        if self.para.arg_representation_method == 'fix_slots':
            # batch x instance_size x event_component
            batch_event_rep = batch_event_data['event_components']
            # batch x context_size x event_component
            batch_context = batch_info['context_events']

            context_emb = self.event_embedding(batch_context)
            event_emb = self.event_embedding(batch_event_rep)

            event_repr = self.arg_composition_model(event_emb)
            context_repr = self.arg_composition_model(context_emb)

            pred_emb = event_emb[:, :, 1, :]
        elif self.para.arg_representation_method == 'role_dynamic':
            batch_event_repr_data = {}
            batch_context_event_repr_data = {}

            # Each value is of shape batch x instance_size,
            # and will become embeddings:
            # batch x instance_size x emb.
            for k in 'predicate', 'frame', 'slot_value', 'slot':
                batch_event_repr_data[k] = self.event_embedding(
                    batch_event_data[k])
                batch_context_event_repr_data[k] = self.event_embedding(
                    batch_info["context_" + k])

            # Some non-embedding items
            # batch x instance_size
            for k in ('slot_length',):
                batch_event_repr_data[k] = batch_event_data[k]
                batch_context_event_repr_data[k] = \
                    batch_info["context_" + k]

            pred_emb = batch_event_repr_data['predicate']

            event_repr = self.arg_composition_model(batch_event_repr_data)
            context_repr = self.arg_composition_model(
                batch_context_event_repr_data)
        else:
            raise ValueError(
                f"Unknown compose method {self.para.arg_representation_method}")

        return pred_emb, event_repr, context_repr

    def forward(self, batch_event_data, batch_info):
        """

        Args:
          batch_event_data: 
          batch_info: 

        Returns:

        """
        # batch x instance_size x n_features
        batch_features = batch_event_data['features']

        # batch x instance_size
        batch_event_indices = batch_info['event_indices']

        # The slot is an embedding the represent the slot role.
        # It can be a one-hot vector for fix slot, and dense embedding in
        # dynamic view.
        # TODO: 1. what should be the indicator? 2. this is currently long,
        #  which won't work
        batch_slots = batch_info['slot_indicators']

        l_extracted = [batch_features, batch_slots]

        # Compute the representation of events and context events.
        pred_emb, event_repr, context_repr = self.event_repr(
            batch_event_data, batch_info)

        # Compute the distance features using the predicate embedding.
        if self._use_distance:
            # batch x instance_size x n_distance_features
            batch_distances = batch_event_data['distances']

            # Adding distance features
            distance_emb = self.distance_module(pred_emb, batch_distances)
            l_extracted.append(distance_emb)

        # Now compute the coherent features with all context events.
        bs, event_size, _ = event_repr.shape
        bs, context_size, _ = context_repr.shape
        self_mask = self.self_event_mask(batch_event_indices, bs, event_size,
                                         context_size)
        coh_features = self.context_vote_layer(event_repr, context_repr,
                                               self_mask)

        l_extracted.append(coh_features)

        pdb.set_trace()

        # batch x instance_size x feature_size
        all_features = torch.cat(l_extracted, -1)

        # batch x instance_size x 1
        scores = self._linear_combine(all_features).squeeze(-1)

        if self.normalize_score:
            scores = torch.nn.Sigmoid()(scores)

        pdb.set_trace()

        return scores
