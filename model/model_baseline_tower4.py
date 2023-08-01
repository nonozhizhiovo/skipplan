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
        self.query_embed2 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init2 = nn.Embedding(args.memory_length, args.d_model)
        self.query_embed3 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init3 = nn.Embedding(args.memory_length, args.d_model)

        self.d_model = args.d_model
        self.H = args.H
        decoder_layer2 = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                args.decoder_dropout, 'relu', normalize_before=True,
                                                memory_size=args.memory_size,
                                                bs=args.batch_size)
        decoder_layer3 = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                args.decoder_dropout, 'relu', normalize_before=True,
                                                memory_size=args.memory_size,
                                                bs=args.batch_size)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder2 = TransformerDecoder(decoder_layer2, args.N, decoder_norm,
                                          return_intermediate=False)
        self.decoder3 = TransformerDecoder(decoder_layer3, args.N, decoder_norm,
                                          return_intermediate=False)
        self.cls_classifier2 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier3 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier = MLP(self.d_model, 106, [args.mlp_mid])

        self.cls_classifier_merge = MLP(8, 4, [24])

        self.feat_reshape = MLP(640, self.d_model, [1024])
        if args.init_weight:
            self.apply(self._init_weights)

        # self.merge = MLP(3, 1, [6])
        self.merge = MLP(3, 1, [3 * args.smallmid_ratio])

    def forward(self, frames):
        # initial and goal

        frames1 = frames.float()
        frames_initial = self.merge(self.feat_reshape(frames1[:, 0])
                                    .transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)
        frames_goal = self.merge(self.feat_reshape(frames1[:, -1])
                                 .transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)

        query_embed2 = self.query_embed2.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed2[0, :] = frames_initial
        query_embed2[-1, :] = frames_goal

        query_embed3 = self.query_embed3.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed3[0, :] = frames_initial
        query_embed3[-1, :] = frames_goal

        # memory
        framesembed2 = self.framesembed_init2.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()
        framesembed3 = self.framesembed_init3.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()

        # net
        out2 = self.decoder2(query_embed2, framesembed2)
        out3 = self.decoder3(query_embed3, framesembed3)

        output2 = out2.transpose(0, 1)[:, :3, :]
        output3 = out3.transpose(0, 1)[:, :3, :]

        input2 = torch.zeros([frames.shape[0], self.max_traj_len, self.d_model]).cuda()
        input3 = torch.zeros([frames.shape[0], self.max_traj_len, self.d_model]).cuda()

        input2[:, 0:2, :] = output2[:, 0:2, :]
        input2[:, 3:4, :] = output2[:, 2:3, :]
        input3[:, 0:1, :] = output3[:, 0:1, :]
        input3[:, 2:4, :] = output3[:, 1:3, :]

        input_merge = torch.cat([input2, input3], dim=1)
        cls_output_merge = self.cls_classifier_merge(input_merge.transpose(-1, -2)).transpose(-1, -2)

        cls_output2 = self.cls_classifier2(output2)
        cls_output3 = self.cls_classifier3(output3)
        cls_output = self.cls_classifier(cls_output_merge)

        return cls_output2, cls_output3, cls_output

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

