from .transformer import *
from utils import *
import math


class Model(nn.Module):
    def __init__(self, args, memory_frames=None, memory_lowlevel_labels=None):
        super(Model, self).__init__()
        self.args = args
        self.frameduration = args.frameduration
        self.adapool = nn.AdaptiveAvgPool1d(1)
        self.max_traj_len = args.max_traj_len
        self.memory_size = args.memory_size
        self.query_embed1 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init1 = nn.Embedding(args.memory_length, args.d_model)
        self.query_embed2 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init2 = nn.Embedding(args.memory_length, args.d_model)
        self.query_embed3 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init3 = nn.Embedding(args.memory_length, args.d_model)
        self.query_embed4 = nn.Embedding(args.query_length, args.d_model)
        self.framesembed_init4 = nn.Embedding(args.memory_length, args.d_model)
        self.d_model = args.d_model
        self.H = args.H
        decoder_layer1 = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                args.decoder_dropout, 'relu', normalize_before=True,
                                                memory_size=args.memory_size,
                                                bs=args.batch_size)
        decoder_layer2 = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                args.decoder_dropout, 'relu', normalize_before=True,
                                                memory_size=args.memory_size,
                                                bs=args.batch_size)
        decoder_layer3 = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                 args.decoder_dropout, 'relu', normalize_before=True,
                                                 memory_size=args.memory_size,
                                                 bs=args.batch_size)
        decoder_layer4 = TransformerDecoderLayer(self.d_model, args.H, args.dim_feedforward,
                                                 args.decoder_dropout, 'relu', normalize_before=True,
                                                 memory_size=args.memory_size,
                                                 bs=args.batch_size)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder1 = TransformerDecoder(decoder_layer1, args.N, decoder_norm,
                                          return_intermediate=False)
        self.decoder2 = TransformerDecoder(decoder_layer2, args.N, decoder_norm,
                                          return_intermediate=False)
        self.decoder3 = TransformerDecoder(decoder_layer3, args.N, decoder_norm,
                                           return_intermediate=False)
        self.decoder4 = TransformerDecoder(decoder_layer4, args.N, decoder_norm,
                                           return_intermediate=False)
        # self.dropout_feas = nn.Dropout(args.feat_dropout)
        self.cls_classifier1 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier2 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier3 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier4 = MLP(self.d_model, 106, [args.mlp_mid])
        self.cls_classifier_merge = MLP(24, 6, [24 * args.smallmid_ratio])
        self.cls_classifier_merge2 = MLP(12, 6, [12 * args.smallmid_ratio])
        self.cls_classifier = MLP(self.d_model, 106, [args.mlp_mid])
        self.feat_reshape = MLP(640, self.d_model, [1024])
        self.apply(self._init_weights)
        self.merge = MLP(3, 1, [3 * args.smallmid_ratio])

    def forward(self, frames):
        frames1 = frames.float()
        frames_initial = self.merge(self.feat_reshape(frames1[:, 0])
                                    .transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)
        frames_goal = self.merge(self.feat_reshape(frames1[:, -1])
                                 .transpose(-1, -2)).transpose(-1, -2).transpose(0, 1)

        query_embed1 = self.query_embed1.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed1[0, :] = frames_initial
        query_embed1[-1, :] = frames_goal

        query_embed2 = self.query_embed2.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed2[0, :] = frames_initial
        query_embed2[-1, :] = frames_goal

        query_embed3 = self.query_embed3.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed3[0, :] = frames_initial
        query_embed3[-1, :] = frames_goal

        query_embed4 = self.query_embed4.weight.unsqueeze(1).repeat(1, frames1.shape[0], 1)
        query_embed4[0, :] = frames_initial
        query_embed4[-1, :] = frames_goal

        # memory
        framesembed1 = self.framesembed_init1.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()
        framesembed2 = self.framesembed_init2.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()
        framesembed3 = self.framesembed_init3.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()
        framesembed4 = self.framesembed_init4.weight.unsqueeze(1).repeat(1, self.memory_size, 1).float()

        # net
        out1 = self.decoder1(query_embed1, framesembed1)
        out2 = self.decoder2(query_embed2, framesembed2)
        out3 = self.decoder3(query_embed3, framesembed3)
        out4 = self.decoder4(query_embed3, framesembed4)

        output1 = out1.transpose(0, 1)[:, :3, :]
        output2 = out2.transpose(0, 1)[:, :3, :]
        output3 = out3.transpose(0, 1)[:, :3, :]
        output4 = out4.transpose(0, 1)[:, :3, :]

        input1 = torch.zeros([frames.shape[0], self.max_traj_len, self.d_model]).cuda()
        input2 = torch.zeros([frames.shape[0], self.max_traj_len, self.d_model]).cuda()
        input3 = torch.zeros([frames.shape[0], self.max_traj_len, self.d_model]).cuda()
        input4 = torch.zeros([frames.shape[0], self.max_traj_len, self.d_model]).cuda()

        input1[:, 0:2, :] = output1[:, 0:2, :]
        input1[:, 5:6, :] = output1[:, 2:3, :]
        input2[:, 0:1, :] = output2[:, 0:1, :]
        input2[:, 2:3, :] = output2[:, 1:2, :]
        input2[:, 5:6, :] = output2[:, 2:3, :]
        input3[:, 0:1, :] = output3[:, 0:1, :]
        input3[:, 3:4, :] = output3[:, 1:2, :]
        input3[:, 5:6, :] = output3[:, 2:3, :]
        input4[:, 0:1, :] = output4[:, 0:1, :]
        input4[:, 4:6, :] = output4[:, 1:3, :]

        input_merge = torch.cat([input1, input2, input3, input4], dim=1)
        cls_output_merge = self.cls_classifier_merge(input_merge.transpose(-1, -2)).transpose(-1, -2)

        cls_output1 = self.cls_classifier1(output1)
        cls_output2 = self.cls_classifier2(output2)
        cls_output3 = self.cls_classifier3(output3)
        cls_output4 = self.cls_classifier4(output4)
        cls_output = self.cls_classifier(cls_output_merge)

        return cls_output1, cls_output2, cls_output3, cls_output4, cls_output

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

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


