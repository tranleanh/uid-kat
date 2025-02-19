# Baseline
class KATGenerator(nn.Module):

    def __init__(self, 
            input_nc, 
            output_nc, 
            ngf, 
            opt=None,
        ):

        super(KATGenerator, self).__init__()

        self.patch_size = opt.patch_size
        self.n_kat_blocks = opt.n_kat_blocks
        self.mixertype = opt.mixer
        self.act_type = opt.act_type
        self.fks = opt.fks
        fpad = (self.fks-1)//2

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(input_nc, ngf, kernel_size=self.fks, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.1),
            # SCBottleneck(ngf, ngf)
        )

        # Downsampleing layer 2 & 3
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.1),
            # SCBottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.LeakyReLU(0.1),
            # SCBottleneck(ngf*4, ngf*4)
        )


        # ======================================================

        # 9 ResNet blocks
        res = []
        for _ in range(9):
            res += [ResidualBlock(ngf*4)]
        self.res = nn.Sequential(*res)

        # ======================================================


        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.LeakyReLU(0.1),
            # SCBottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.1),
            # SCBottleneck(ngf, ngf)
        )

        # Output layer (original)
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(ngf*2, output_nc, self.fks),
            nn.Tanh()
        )


    def forward(self, input, layers=[], encode_only=False):

        layers1 = [0]
        layers2 = [0, 4]

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.conv_1):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_2):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_3):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.res):
                feat = layer(feat)
                if layer_id in layers2:
                    feats.append(feat)
                else:
                    pass
            if encode_only:
                return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            # Encoding
            # batch size x 64 x 256 x 256
            c1 = self.conv_1(input)

            # batch size x 128 x 128 x 128
            c2 = self.conv_2(c1)

            # batch size x 256 x 64 x 64
            c3 = self.conv_3(c2)

            # KAT blocks
            c4 = self.res(c3)

            # Decoding
            # batch size x 512 x 64 x 64
            skip1_de = torch.cat((c3, c4), 1)

            # batch size x 128 x 128 x 128
            c1_de = self.conv_5(skip1_de)

            # batch size x 256 x 128 x 128
            skip2_de = torch.cat((c2, c1_de), 1)

            # batch size x 64 x 256 x 256
            c3_de = self.conv_6(skip2_de)

            # batch size x 128 x 256 x 256
            skip3_de = torch.cat((c1, c3_de), 1)

            # batch size x 3 x 256 x 256
            fake = self.conv_7(skip3_de)
            return fake





# + SCBottleneck + SiLU
class KATGenerator(nn.Module):

    def __init__(self, 
            input_nc, 
            output_nc, 
            ngf, 
            opt=None,
        ):

        super(KATGenerator, self).__init__()

        # self.opt = opt
        # self.patch_size = patch_size
        self.patch_size = opt.patch_size
        self.n_kat_blocks = opt.n_kat_blocks
        self.mixertype = opt.mixer
        self.act_type = opt.act_type
        # self.iokan = opt.iokan
        self.fks = opt.fks
        fpad = (self.fks-1)//2

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(input_nc, ngf, kernel_size=self.fks, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.SiLU(),
            SCBottleneck(ngf, ngf)
        )

        # Downsampleing layer 2 & 3
        # inputsize:64*256*256, outputsize:128*128*128
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.SiLU(),
            SCBottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:256*64*64
        self.conv_3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.SiLU(),
            SCBottleneck(ngf*4, ngf*4)
        )


        # ======================================================

        # 9 ResNet blocks
        res = []
        for _ in range(9):
            res += [ResidualBlock(ngf*4)]
        self.res = nn.Sequential(*res)

        # ======================================================


        # Upsampling
        # inputsize:256*64*64, outputsize:128*128*128
        self.conv_5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.SiLU(),
            SCBottleneck(ngf*2, ngf*2)
        )

        # inputsize:128*128*128, outputsize:64*256*256
        self.conv_6 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.SiLU(),
            SCBottleneck(ngf, ngf)
        )

        # Output layer (original)
        self.conv_7 = nn.Sequential(
            nn.ReflectionPad2d(fpad),
            nn.Conv2d(ngf*2, output_nc, self.fks),
            nn.Tanh()
        )


    def forward(self, input, layers=[], encode_only=False):

        layers1 = [0]
        layers2 = [0, 4]

        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.conv_1):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_2):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.conv_3):
                feat = layer(feat)
                if layer_id in layers1:
                    feats.append(feat)
                else:
                    pass

            for layer_id, layer in enumerate(self.res):
                feat = layer(feat)
                if layer_id in layers2:
                    feats.append(feat)
                else:
                    pass
            if encode_only:
                return feats  # return intermediate features alone; stop in the last layers
        else:
            """Standard forward"""
            # Encoding
            # batch size x 64 x 256 x 256
            c1 = self.conv_1(input)

            # batch size x 128 x 128 x 128
            c2 = self.conv_2(c1)

            # batch size x 256 x 64 x 64
            c3 = self.conv_3(c2)

            # KAT blocks
            c4 = self.res(c3)

            # Decoding
            # batch size x 512 x 64 x 64
            skip1_de = torch.cat((c3, c4), 1)

            # batch size x 128 x 128 x 128
            c1_de = self.conv_5(skip1_de)

            # batch size x 256 x 128 x 128
            skip2_de = torch.cat((c2, c1_de), 1)

            # batch size x 64 x 256 x 256
            c3_de = self.conv_6(skip2_de)

            # batch size x 128 x 256 x 256
            skip3_de = torch.cat((c1, c3_de), 1)

            # batch size x 3 x 256 x 256
            fake = self.conv_7(skip3_de)
            return fake





class KATBlock(nn.Module):
    def __init__(self, 
            patch_size=8, 
            in_chans=256, 
            out_chans=256, 
            embed_dim=256,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KAN,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            proj_drop: float = 0.,
            act_init: str = 'gelu',
            num_heads=8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            use_attn: bool = False,
        ):
        super(KATBlock, self).__init__()

        self.patchemd = PatchEmbed(patch_size, in_chans, embed_dim)

        self.norm1 = norm_layer(embed_dim)

        if use_attn:
            self.mixer = Attention(
                embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )                                  # <====== Token Mixer: Replace with Pooling
        else:
            self.mixer = nn.Identity()

        self.norm2 = norm_layer(embed_dim)


        # self.mlp = mlp_layer(
        #     in_features=embed_dim,
        #     hidden_features=int(embed_dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        #     act_init=act_init,
        # )                                    # <====== MLP: Replace with KAT

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(True),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

        self.patchunemb = PatchUnEmbed(patch_size, out_chans, embed_dim)

    def forward(self, x):

        x = self.patchemd(x)
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.patchunemb(x)
        return x