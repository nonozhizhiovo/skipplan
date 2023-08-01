from .transformer import *
from utils import *

class Model(nn.Module):
    def __init__(self, args, memory_frames=None, memory_lowlevel_labels=None):
        super(Model, self).__init__()
        self.args = args
        self.frameduration = args.frameduration
        self.adapool = nn.AdaptiveAvgPool1d(1)
        self.max_traj_len = args.max_traj_len
        self.memory_size = args.memory_size
        self.query_length = args.query_length
        self.memory_length = args.memory_length
        self.query_embed = nn.Embedding(self.query_length, args.d_model)
        self.framesembed_init = nn.Embedding(self.memory_length, args.d_model)
        self.d_model = args.d_model
        self.H = args.H
        decoder_layer = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                args.decoder_dropout, 'relu', normalize_before=True,
                                                memory_size=args.memory_size,
                                                bs=args.batch_size)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, args.N, decoder_norm,
                                          return_intermediate=False)
        # self.dropout_feas = nn.Dropout(args.feat_dropout)
        self.cls_classifier = MLP(self.d_model, 106, [args.mlp_mid])
        self.feat_reshape = MLP(640, self.d_model, [1024])
        self.apply(self._init_weights)
        # self.merge = MLP(3, 1, [6])
        self.merge = MLP(3, 1, [3 * args.smallmid_ratio])

    def forward(self, frames):
        # initial and goal
        frames1 = frames.float()
        frames_initial = self.merge(self.feat_reshape(frames1[:, 0]).transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)
        frames_goal = self.merge(self.feat_reshape(frames1[:, -1]).transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed[0, :] = frames_initial
        query_embed[-1, :] = frames_goal

        # memory
        framesembed = self.framesembed_init.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()

        # net
        out = self.decoder(query_embed, framesembed)
        # out = self.dropout_feas(out)
        cls_output = self.cls_classifier(out).transpose(0, 1)[:, :self.max_traj_len, :]
        return cls_output

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


