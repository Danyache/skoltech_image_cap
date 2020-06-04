import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch
from torch import nn
import torchvision
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch.nn.functional as F
from pathlib import Path

from collections import OrderedDict

curr_path = Path(os.getcwd())

NLMCXR_path = os.path.join(str(curr_path.parent), 'NLMCXR_data')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings_glove = torch.load(f"{NLMCXR_path}/embeds/embeddings_glove_v10.pt")

MODEL = "DENSE" # "resnet" #

if MODEL == "DENSE":
    ENC_DIM = 1024
else:
    ENC_DIM = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class DenseNet121(nn.Module):
	"""Model modified.
	The architecture of our model is the same as standard DenseNet121
	except the classifier layer which has an additional sigmoid function.
	"""
	def __init__(self, out_size):
		super(DenseNet121, self).__init__()
		self.densenet121 = torchvision.models.densenet121(pretrained=True)
		num_ftrs = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(
		    nn.Linear(num_ftrs, out_size),
		    nn.Sigmoid()
		)

	def forward(self, x):
		x = self.densenet121(x)
		return x

good_layers = ["densenet121.features.conv0.weight", "densenet121.features.norm0.weight", "densenet121.features.norm0.bias", "densenet121.features.norm0.running_mean", "densenet121.features.norm0.running_var", "densenet121.features.transition1.norm.weight", "densenet121.features.transition1.norm.bias", "densenet121.features.transition1.norm.running_mean", "densenet121.features.transition1.norm.running_var", "densenet121.features.transition1.conv.weight", "densenet121.features.transition2.norm.weight", "densenet121.features.transition2.norm.bias", "densenet121.features.transition2.norm.running_mean", "densenet121.features.transition2.norm.running_var", "densenet121.features.transition2.conv.weight", "densenet121.features.transition3.norm.weight", "densenet121.features.transition3.norm.bias", "densenet121.features.transition3.norm.running_mean", "densenet121.features.transition3.norm.running_var", "densenet121.features.transition3.conv.weight", "densenet121.features.norm5.weight", "densenet121.features.norm5.bias", "densenet121.features.norm5.running_mean", "densenet121.features.norm5.running_var", "densenet121.classifier.0.weight", "densenet121.classifier.0.bias"]

class CrossGPTLayer(nn.Module):
    """
    CrossLayer
    """
    def __init__(self, word_map):
        super(CrossGPTLayer, self).__init__()
        self.vocab = word_map
        self.rev_vocab = {v: k for k, v in word_map.items()}
        
        self.voc_len = len(self.vocab)
        
        self.cross = nn.Linear(self.voc_len * 2, 1024)
        self.cross2 = nn.Linear(1024, self.voc_len)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(f'{NLMCXR_path}/output3')
        self.model = GPT2LMHeadModel.from_pretrained(f'{NLMCXR_path}/output3', pad_token_id=self.tokenizer.eos_token_id).to(device)
        
         
        
        self.vocab_gpt = self.tokenizer.get_vocab()
        self.rev_vocab_gpt = {i:v for v, i in self.vocab_gpt.items()}
        
        gpt_indexes = []
        err_tokens = []

        for token in word_map.keys():
            try:
                gpt_indexes.append(self.vocab_gpt[token])
            except:
                gpt_indexes.append(self.vocab_gpt['<|endoftext|>'])
                err_tokens.append(token)
        
        self.gpt_indexes = gpt_indexes
        
    def forward(self, predictions, encoded_captions, decode_lengths):
        # print(predictions.shape, encoded_captions.shape, decode_lengths[0])
            
        gpt_preds = self.idxs2gpt_predictions(encoded_captions[:, :decode_lengths[0]])
        # print(gpt_preds.shape)
        
        new_preds = self.cross(torch.cat((predictions, gpt_preds), dim=2))
        
        
        return self.cross2(F.relu(new_preds))
        
    def idxs2gpt_predictions(self, batch_idx):
        predictions = torch.ones((batch_idx.shape[0], batch_idx.shape[1], len(self.vocab))).to(device)

        for j, idx in enumerate(batch_idx):
            tokens = [self.rev_vocab[int(i)] for i in idx]
            torch_tokens = torch.tensor(self.tokenizer.encode(tokens)).unsqueeze(0).to(device)
            predictions[j, :, :] = self.model(torch_tokens)[0][0][:, self.gpt_indexes]

        return predictions
        
        
        

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        
        inception = DenseNet121(8).cuda() # inception_v3_base(pretrained=True)
        state_dict = torch.load("/raid/data/cxr14-2/DenseNet121_aug4_pretrain_WeightBelow1_1_0.829766922537.pkl")
        new_state_dict = OrderedDict()

        for s, v in state_dict.items():

            if 'module.' in s:
                s = s.replace('module.', '')

            if s not in good_layers:
                s = '.'.join(s.split('.')[:-2]) + '.'.join(s.split('.')[-2:])

            new_state_dict[s] = v
        inception.load_state_dict(new_state_dict)

        self.dense_m = inception._modules['densenet121']
        
        
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        
        # if MODEL != "DENSE":
        # self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if MODEL == "DENSE":
            for f in self.dense_m.features:
                images = f(images)
            outs = images.permute(0, 2, 3, 1) #.view(-1, 49, 1024)
        else:
            out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
            out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            outs = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
            
        return outs

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=ENC_DIM, dropout=0.5, embeds=embeddings_glove):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        # self.embedding = nn.Embedding(embeds.shape[0], embed_dim)
        # print("!!!", self.embedding, "!!!")
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        # self.embedding = nn.Embedding.from_pretrained(embeds)
        
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
