import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

N_INSTRUMENTS = 13


class ResNetLSTMFeatures(nn.Module):

    def __init__(self, with_resnet=False, with_lstm=True, max_pooling=True,
                 visual_finetune=False, get_probabilities=False):
        """

        :param checkpoint_path: path to saved weights for pretrained ResNet50 model
        :param finetune: allow gradient to propagate through visual model as well
        """
        super(ResNetLSTMFeatures, self).__init__()

        self.hidden_size = 1024
        self.with_resnet = with_resnet
        self.with_lstm = with_lstm
        self.max_pooling = max_pooling
        self.visual_finetune = visual_finetune
        self.get_probabilities = get_probabilities
        self.probabilities = None
        if self.with_resnet:
            self.model = torchvision.models.resnet50(pretrained=True)
            # replace last fully-connected layer with identity layer to obtain features
            self.model.fc = nn.Identity()
        if self.with_lstm:
            self.LSTM = nn.LSTM(input_size=2048, hidden_size=self.hidden_size, batch_first=True, bidirectional=False)
        if self.get_probabilities:
            self.fc = nn.Sequential(
                nn.Linear(self.hidden_size, N_INSTRUMENTS),
                nn.Softmax()
            )

    def forward(self, x):
        # input of size (bs x ) k x n_frames x 3 x 224 x 224 if self.with_resnet
        # else bs x k x n_frames x 2048

        (B, K, T, C, H, W) = x.shape
        x = x.view(B * K * T, C, H, W)
        if self.with_resnet:
            if self.visual_finetune:
                x = self.model(x)
            else:
                with torch.no_grad():
                    x = self.model(x)
        x = x.view(B, K * T, x.shape[-1])
        if self.with_lstm:
            lstm_output, (ht, ct) = self.LSTM(x)
            context = ht[-1]
        else:
            # if not lstm then spatio-temporal max pooling:
            context = F.adaptive_max_pool1d(x, self.hidden_size)
            context, _ = torch.max(context, dim=1)

        if self.get_probabilities:
            context = self.fc(context)
            self.probabilities = context.detach().data.cpu().numpy()

        return context
