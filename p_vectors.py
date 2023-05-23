"""A popular speaker recognition and diarization model.

Authors
 * Hwidong Na 2020
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    """1D convolution. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    """1D batch normalization. Skip transpose is used to improve efficiency."""

    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
        The groups size of the TDNN blocks.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            activation=nn.ReLU,
            groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
            self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> se_layer = SEBlock(64, 16, 64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = se_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""

        s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            res2net_scale=8,
            se_channels=128,
            kernel_size=1,
            dilation=1,
            activation=torch.nn.ReLU,
            groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)

        return x + residual

class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self,mode):
        super(SpatialAttentionModul, self).__init__()
        self.mode = mode
        self.conv = nn.Conv1d(3, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.mode == 't2c':
            # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
            maxp = torch.max(x, dim=2, keepdim=True).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
            avgp = torch.mean(x, dim=2, keepdim=True)
            stdd = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4))

            # 维度拼接 [N, 2, H, W]
            x_cat = torch.cat((maxp, avgp,stdd), dim=2).transpose(1,2) # 获得特征图

            # 卷积操作得到空间注意力结果
            x_cat = self.conv(x_cat).transpose(1,2)
            Ms = self.sigmoid(x_cat)

            # 与原图通道进行乘积
            x = Ms * x

            return x
        else:
            maxp = torch.max(x, dim=1, keepdim=True).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
            avgp = torch.mean(x, dim=1, keepdim=True)
            stdd = torch.sqrt(torch.var(x, dim=1, keepdim=True).clamp(min=1e-4))
            x_cat = torch.cat((maxp, avgp, stdd), dim=1)  # 获得特征图

            x_cat = self.conv(x_cat)
            Ms = self.sigmoid(x_cat)

            # 与原图通道进行乘积
            x = Ms * x

            return x

class Repres(torch.nn.Module):

    def __init__(self, enlarge=80):
        super().__init__()

        self.enlarge1 = nn.Conv2d(1, enlarge, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(enlarge)
        self.re1 = nn.ReLU()

        # self.f_avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        # self.f_down = nn.Linear(80,40,bias=False)
        # self.f_relu = nn.ReLU()
        # self.f_up = nn.Linear(40, 80, bias=False)
        # self.f_sigmoid = nn.Sigmoid()
        #
        # self.c_avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        # self.c_down = nn.Linear(enlarge,8,bias=False)
        # self.c_relu = nn.ReLU()
        # self.c_up = nn.Linear(8,enlarge, bias=False)
        # self.c_sigmoid = nn.Sigmoid()


        self.conv = nn.Conv2d(2,1,kernel_size = 5, padding=2,bias=False)
        self.conv_sigmoid = nn.Sigmoid()

        self.enlarge2 = nn.Conv2d(enlarge, 1, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(1)
        self.re2 = nn.ReLU()
    def forward(self,x):

        x = x.unsqueeze(1)

        x = self.re1(self.bn1(self.enlarge1(x)))

        b, c, f, t = x.shape

        # c_coeff = self.c_sigmoid(self.c_up(self.c_relu(self.c_down(self.c_avg_pooling(x).view([b,16]))))).view([b,16,1,1]) # b c 1 1
        #
        # f_coeff = self.f_sigmoid(self.f_up(self.f_relu(self.f_down(self.f_avg_pooling(x.transpose(1,2)).view([b,80]))))).view([b,1,80,1])
        #
        # x = x * c_coeff *f_coeff

        x_maxpool,_ = torch.max(x,dim = 3,keepdim=True)
        x_avgpool= torch.mean(x, dim=3, keepdim=True)

        x_cat = torch.cat([x_maxpool,x_avgpool],dim=3)

        x_cat = self.conv_sigmoid(self.conv(x_cat.permute(0,3,1,2)).permute(0,2,3,1))

        x = x*x_cat
        #
        x = self.re2(self.bn2(self.enlarge2(x)))

        x = x.squeeze(1)

        return x


class R2C(torch.nn.Module):
    def __init__(self):
        super(R2C,self).__init__()
        self.bridge = nn.Conv1d(256, 512, kernel_size=3, stride=1,padding=1)
        # self.bridge = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=0)
        self.ac = nn.BatchNorm1d(512)
        # self.p = nn.Parameter(torch.rand(512))
        # self.p = nn.Parameter(
        #     torch.FloatTensor(1, 512,1)
        # )
        # nn.init.xavier_uniform_(self.p)

    def forward(self,x):
        x = F.interpolate(x.transpose(1,2),size=x.size(1)*3)
        x = self.bridge(x)
        x = self.ac(x)
        # print(x.shape)
        # x = self.p*x
        return x


class C2R(torch.nn.Module):
    def __init__(self):
        super(C2R, self).__init__()
        self.bridge = nn.Conv1d(512, 256, kernel_size=3, stride=3)
        self.ac = nn.LayerNorm(256)
        # self.p = nn.Parameter(
        #     torch.FloatTensor(1, 1, 256)
        # )
        # nn.init.xavier_uniform_(self.p)

    def forward(self, x):
        x = self.bridge(x).transpose(1,2)
        x = self.ac(x)

        # x = self.p*x
        return x

class P_vectors(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
            self,
            input_size,
            device="cpu",
            lin_neurons=192,
            activation=torch.nn.ReLU,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            groups=[1, 1, 1, 1, 1],
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.enlarge1 = Repres()
        # The initial TDNN layer
        self.layer1 = TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],)



        self.SER1 =SERes2NetBlock(
                    channels[0],
                    channels[1],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[1],
                    dilation=dilations[1],
                    activation=activation,
                    groups=groups[1],)

        self.SER2 = SERes2NetBlock(
            channels[1],
            channels[2],
            res2net_scale=res2net_scale,
            se_channels=se_channels,
            kernel_size=kernel_sizes[2],
            dilation=dilations[2],
            activation=activation,
            groups=groups[2], )

        self.SER3 = SERes2NetBlock(
            channels[2],
            channels[3],
            res2net_scale=res2net_scale,
            se_channels=se_channels,
            kernel_size=kernel_sizes[3],
            dilation=dilations[3],
            activation=activation,
            groups=groups[3], )

        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,)

        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,)

        self.enlarge1_t = Repres()

        self.layer1_t = TDNNBlock(
            input_size,
            channels[0],
            kernel_sizes[0],
            dilations[0],
            activation,
            groups[0], )

        self.encoder1 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder2 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder3 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder4 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder5 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder6 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder7 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder8 = nn.TransformerEncoderLayer(256, 8, dim_feedforward=512, dropout=0.1, activation='gelu')

        self.makepatch = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3, padding=0)

        self.mfa_t = TDNNBlock(
            256 * 3,
            256 * 3,
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp_t = AttentiveStatisticsPooling(
            256 * 3,
            attention_channels=attention_channels,
            global_context=global_context, )

        self.asp_bn_t = BatchNorm1d(input_size=512 * 3)

        # Final linear transformation
        self.fc_t = Conv1d(
            in_channels=512 * 3,
            out_channels=192,
            kernel_size=1, )
########################################################
# commont these two lines out duaring training stage 2
# uncommont these two lines duaring training stage 3
        # for p in self.parameters():  
        #     p.requires_grad = False
########################################################
        self.fc_all = Conv1d(in_channels=384,
            out_channels=192,
            kernel_size=1)

        self.r2c1 = R2C()
        self.c2r2 = C2R()
        self.r2c3 = R2C()
        self.c2r4 = C2R()




    def forward(self, x, lengths=None):

        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        x = x[:, :, :3 * (x.size(-1) // 3)]

        # print(x.shape)

        x_t = self.enlarge1_t(x)

        # print(x_t.shape)

        x_t = self.layer1_t(x_t)

        # print(x_t.shape)
        #
        # print('*******************')

        x1 = self.enlarge1(x)

        x1 = self.layer1(x1)

        x1 = self.SER1(x1)

        x_t_1 = self.encoder2(self.encoder1(self.makepatch(x_t).transpose(1,2))) #

        x2 = self.SER2(x1+self.r2c1(x_t_1))

        x_t_2 = self.encoder5(self.encoder4(self.encoder3(self.c2r2(x2)+x_t_1)))

        x3 = self.SER3(x2 + self.r2c3(x_t_2))

        x_t_3 = self.encoder8(self.encoder7(self.encoder6(self.c2r4(x3) + x_t_2)))

        # Multi-layer feature aggregation
        x = torch.cat((x1,x2,x3), dim=1)
        # print(x.shape)
        # print(x.shape)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)
        # print(x_t_1.shape)
        x_t = torch.cat((x_t_1,x_t_2,x_t_3),dim=2).transpose(1,2)

        # print(x_t.shape)
        x_t = self.mfa_t(x_t)
        x_t = self.asp_t(x_t,lengths=lengths)
        x_t = self.asp_bn_t(x_t)
        x_t = self.fc_t(x_t)


        x_all = torch.cat((x,x_t),dim=1)
        x_all = self.fc_all(x_all).transpose(1,2)
        # print(x.shape,x_t.shape)
        return x_all


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.

    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
            self,
            input_size,
            device="cpu",
            lin_blocks=0,
            lin_neurons=192,
            out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()


        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)





    def forward(self, x):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """

        for layer in self.blocks:
            x = layer(x)

        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))

        return x.unsqueeze(1)


