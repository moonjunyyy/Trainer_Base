import torch

class Ouroboros(torch.nn.Module):
    def __init__(self, input_size = 224, hidden_size = 768, ffn_size = 3072,  num_classes=100, patch_size=16, num_encoder_layer=12, num_decoder_layer=6, dropout=0.1):
        super(Ouroboros, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layer = num_decoder_layer
        self.dropout = dropout


        self.embeding = torch.nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        self.cls_tocken = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.msk_token = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.pos_embeding = torch.nn.Parameter(torch.randn(1, (input_size//patch_size)**2+1, hidden_size), requires_grad=True)


        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, 12, ffn_size, dropout, 'gelu'),
            num_encoder_layer,
            torch.nn.LayerNorm(hidden_size),
            None
        )
        self.decoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, 12, ffn_size, dropout, 'gelu'),
            num_decoder_layer,
            torch.nn.LayerNorm(hidden_size),
            None
        )
        self.head = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.embeding(x)
        x = x.flatten(2).permute(0, 2, 1)
        B, N, C = x.size()
        img = x.clone().detach()

        x = torch.cat([self.cls_tocken.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embeding

        x = self.encoder(x)
        logit = self.head(x[:,0].clone())

        # indices = torch.argsort(torch.rand(B, N), dim=-1) + 1
        # masksize = N - int(N*0.25)
        # masked = indices[:, int(N*0.25):]
        # unmasked = indices[:, :int(N*0.25)]

        
        # x = torch.cat([x[:,:1].clone(), x[torch.arange(B).unsqueeze(-1), unmasked].clone(), self.msk_token.expand(B, masksize, -1) + self.pos_embeding.expand(B, -1, -1)[torch.arange(B).unsqueeze(-1), masked]], dim=1)
        # x[torch.arange(B).unsqueeze(-1), mask] *= 0
        # x[torch.arange(B).unsqueeze(-1), mask] += self.msk_token

        # x = self.decoder(x)
        # self.reconstruct_loss = torch.nn.functional.mse_loss(x[:,1:], img[torch.arange(B).unsqueeze(-1), indices])

        return logit
    
    def loss_fn(self, logit, label):
        return torch.nn.functional.cross_entropy(logit, label) # + self.reconstruct_loss
