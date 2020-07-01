import torch
import torch.nn.functional as F

from megatron import get_args, mpu
from megatron.model.bert_model import (BertLMHead, bert_attention_mask_func,
                                       bert_extended_attention_mask,
                                       bert_position_ids)
from megatron.model.utils import (get_linear_layer, init_method_normal,
                                  openai_gelu, scaled_init_method_normal)
from megatron.module import MegatronModule
from megatron.model.transformer import ParallelTransformerLayer
# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm
from megatron.model.language_model import Embedding, Pooler


class LayerwiseParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, attention_mask_func, mlp_activation_func,
                 init_method, output_layer_init_method):
        super(LayerwiseParallelTransformer, self).__init__()
        args = get_args()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers:
        self.num_layers = args.num_layers
        self.num_unique_layers = args.num_unique_layers
        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert self.num_layers % self.num_unique_layers == 0, \
            'number of layers should be divisible by number of unique layers'
        self.param_sharing_style = args.param_sharing_style

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                attention_mask_func, mlp_activation_func,
                init_method, output_layer_init_method, layer_number)
        # self.layers = torch.nn.ModuleList(
        #     [build_layer(i + 1) for i in range(self.num_unique_layers)])
        self.layers = []
        self.layer_devices = []
        assert self.num_layers % args.n_gpus == 0, 'number of layers should be divisible by number of GPUs'
        layer_per_gpu = self.num_layers // args.n_gpus
        _c = 1
        for i in range(args.n_gpus):
            # move the layers to corresponding GPU
            self.layers.extend([build_layer(_c).to("cuda:"+str(i)) for _ in range(layer_per_gpu)])
            self.layer_devices.extend([i for _ in range(layer_per_gpu)])
            _c += 1
        self.layers = torch.nn.ModuleList(self.layers)

        # Print layer ordering.
        if self.num_layers != self.num_unique_layers:
            if torch.distributed.get_rank() == 0:
                print('> will be using the following layer ordering:')
                for i in range(self.num_layers):
                    print('   layer id: {:3d} --> unique layer id: '
                          '{:3d}'.format(i, self._get_layer_index(i)),
                          flush=True)

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon).to("cuda:"+str(args.n_gpus-1))

    def _get_layer_index(self, layer_number):
        if self.param_sharing_style == 'grouped':
            return layer_number % self.num_unique_layers
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.num_layers // self.num_unique_layers) 
        assert False, 'should not be here'

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x2_ = inputs[1]
                    if x_.device.index != self.layer_devices[index]:
                        x_ = x_.to("cuda:"+ str(self.layer_devices[index]))
                    if x2_.device.index != self.layer_devices[index]:
                        x2_ = x2_.to("cuda:"+ str(self.layer_devices[index]))
                    x_ = layer(x_, x2_)
                return x_
            return custom_forward

        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):

        # Checks
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        if self.checkpoint_activations:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                past = None
                if layer_past is not None:
                    past = layer_past[index]
                if index == 0 or self.layer_devices[index - 1] != self.layer_devices[index]:
                    hidden_states = hidden_states.to("cuda:"+ str(self.layer_devices[index]))
                    attention_mask = attention_mask.to("cuda:"+ str(self.layer_devices[index]))
                
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if get_key_value:
            output = [output, presents]

        return output


class LayerwiseTransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 attention_mask_func,
                 mlp_activation_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):
        super(LayerwiseTransformerLanguageModel, self).__init__()
        args = get_args()
        self.args = args

        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_pooler = add_pooler

        # Embeddings
        self.embedding = Embedding(self.hidden_size,
                                   args.padded_vocab_size,
                                   args.max_position_embeddings,
                                   args.hidden_dropout,
                                   self.init_method,
                                   self.num_tokentypes).to('cuda:0')
        self._embedding_key = 'embedding'

        # Transformer
        self.transformer = LayerwiseParallelTransformer(
            attention_mask_func, mlp_activation_func,
            self.init_method, output_layer_init_method)
        self._transformer_key = 'transformer'

        # Pooler
        if self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method).to('cuda:'+str(args.n_gpus-1))
            self._pooler_key = 'pooler'

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                pooling_sequence_index=0):
        # input_ids = input_ids.to('cuda:' + str(self.args.n_gpus - 1))
        # position_ids = position_ids
        # tokentype_ids = tokentype_ids.to('cuda:' + str(self.args.n_gpus - 1))
        # Embeddings.
        embedding_output = self.embedding(input_ids, position_ids,
                                          tokentype_ids=tokentype_ids)

        # Transformer.
        transformer_output = self.transformer(embedding_output,
                                              attention_mask,
                                              layer_past=layer_past,
                                              get_key_value=get_key_value)

        if self.add_pooler:
            pooled_output = self.pooler(transformer_output,
                                        pooling_sequence_index)
            return transformer_output, pooled_output

        return transformer_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._embedding_key] \
            = self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._transformer_key] \
            = self.transformer.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.add_pooler:
            state_dict_[self._pooler_key] \
                = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self._embedding_key in state_dict:
            state_dict_ = state_dict[self._embedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.load_state_dict(state_dict_, strict=strict)

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]
        self.transformer.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.add_pooler:
            assert 'pooler' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.pooler.load_state_dict(state_dict[self._pooler_key],
                                        strict=strict)


def get_language_model(attention_mask_func, num_tokentypes, add_pooler,
                       init_method, scaled_init_method):
    """Build language model and return along with the key to save."""
    args = get_args()

    # Use torch gelu unless otherwise forced.
    gelu = F.gelu
    if args.openai_gelu:
        gelu = openai_gelu

    # Language model.
    language_model = LayerwiseTransformerLanguageModel(
        attention_mask_func=attention_mask_func,
        mlp_activation_func=gelu,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        num_tokentypes=num_tokentypes,
        add_pooler=add_pooler)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class LayerwiseBertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self, num_tokentypes=2, add_binary_head=True,
                 parallel_output=True):
        super(LayerwiseBertModel, self).__init__()
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            init_method=init_method,
            scaled_init_method=scaled_init_method)

        self.lm_head = BertLMHead(
            self.language_model.embedding.word_embeddings.weight.size(0),
            args.hidden_size, init_method, args.layernorm_epsilon,
            parallel_output).to('cuda:0')
        self._lm_head_key = 'lm_head'

        if self.add_binary_head:
            self.binary_head = get_linear_layer(args.hidden_size, 2,
                                                init_method)
            self.binary_head = self.binary_head.to('cuda:0')
            self._binary_head_key = 'binary_head'

    def forward(self, input_ids, attention_mask,
                tokentype_ids=None, lm_labels=None):

        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)

        if self.add_binary_head:
            lm_output, pooled_output = self.language_model(
                input_ids,
                position_ids,
                extended_attention_mask,
                tokentype_ids=tokentype_ids)
            lm_output, pooled_output = lm_output.to('cuda:0'), pooled_output.to('cuda:0')
        else:
            lm_output = self.language_model(
                input_ids,
                position_ids,
                extended_attention_mask,
                tokentype_ids=tokentype_ids)
            lm_output = lm_output.to('cuda:0')

        # Output.
        lm_logits = self.lm_head(
            lm_output, self.language_model.embedding.word_embeddings.weight)

        binary_logits = None
        if self.add_binary_head:
            binary_logits = self.binary_head(pooled_output)

        if lm_labels is None:
            return lm_logits, binary_logits
        else:
            if self.fp16_lm_cross_entropy:
                assert lm_logits.dtype == torch.half  # pylint: disable=no-member
                lm_loss = mpu.vocab_parallel_cross_entropy(
                    lm_logits, lm_labels)
            else:
                lm_loss = mpu.vocab_parallel_cross_entropy(lm_logits.float(),
                                                           lm_labels)
            return lm_loss, binary_logits

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        state_dict_[self._lm_head_key] \
            = self.lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        self.lm_head.load_state_dict(state_dict[self._lm_head_key],
                                     strict=strict)
        if self.add_binary_head:
            self.binary_head.load_state_dict(state_dict[self._binary_head_key],
                                             strict=strict)
