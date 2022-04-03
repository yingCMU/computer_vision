import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self,  embed_size, hidden_size, hidden_size_2,vocab_size, num_layers=1, debug=False):
        super(DecoderRNN, self).__init__()
        self.debug = debug
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_size_2=hidden_size_2
        self.hidden=None

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # the linear layer that maps the hidden state output dimension
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        if self.hidden_size_2 is not None:
            self.hidden2vocab_1 = nn.Linear(hidden_size, hidden_size_2)
            self.hidden2vocab_2 = nn.Linear(hidden_size_2, vocab_size)
        else:
            self.hidden2vocab_1 = nn.Linear(hidden_size, vocab_size)


    def forward(self,  features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        self.print_debug('before concat ; features=',features.shape)
        self.print_debug('before concat ; captions=',captions.shape)
        captions_embeds = self.word_embeddings(captions)
        self.print_debug('after embedding: captions_embeds =',captions_embeds.shape)
        new_features=torch.unsqueeze(features, 1)
        cat_inputs = torch.cat((new_features, captions_embeds), 1)
        self.print_debug('after concat inputs=', cat_inputs.shape)
        last_idx=cat_inputs.size(1)-1
        removed_endword = cat_inputs[:, torch.arange(cat_inputs.size(1))!=last_idx, :]
        self.print_debug('after remove end word removed_endword=', removed_endword.shape)

        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        # N x (cap_zie +1) x embeding_size
        self.hidden=None
        lstm_out, self.hidden = self.lstm(
            removed_endword, self.hidden)
        self.print_debug('lstm_out=',lstm_out.shape)
        self.print_debug('cell state=',self.hidden[0].shape)
        self.print_debug('hidden state=',self.hidden[1].shape)
        # get the scores for the most likely tag for a word
        if self.hidden_size_2 is not None:
            vocab_outputs_1 = F.relu(self.hidden2vocab_1(lstm_out))
            vocab_outputs = self.hidden2vocab_2(vocab_outputs_1)
        else:
             vocab_outputs = self.hidden2vocab_1(lstm_out)
        self.print_debug('after hidden2vocab: vocab_outputs=', vocab_outputs.shape)
#         vocab_scores = F.log_softmax(vocab_outputs, dim=2)
#         self.print_debug('after log_softmax: vocab_scores=', vocab_scores.shape)
#         output= torch.argmax(vocab_scores, dim=2)
#         self.print_debug('after argmax: output=', output.shape)
        return vocab_outputs

    def print_debug(self, *args):
        if self.debug:
            print(args)


    def sample(self, image_features, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        self.eval()
        self.hidden = None
        predictions=[]
        intputs = image_features
        self.print_debug('image_features shape=', image_features.shape)
        # Now pass in the previous character and get a new one
        caption_output = None
        end_tensor = torch.Tensor([1])
        for ii in range(max_len):
            if caption_output is not None:
                  intputs =self.word_embeddings(caption_output)
            lstm_out, self.hidden = self.lstm(intputs, self.hidden)
            if self.hidden_size_2 is not None:
                vocab_outputs_1 = F.relu(self.hidden2vocab_1(lstm_out))
                vocab_outputs = self.hidden2vocab_2(vocab_outputs_1)
            else:
                 vocab_outputs = self.hidden2vocab_1(lstm_out)
            self.print_debug('vocab_outputs shape=', vocab_outputs.shape)
            vocab_scores = F.log_softmax(vocab_outputs, dim=2)
            self.print_debug('after log_softmax: vocab_scores=', vocab_scores.shape)
            caption_output= torch.argmax(vocab_scores, dim=2)
            self.print_debug('after argmax output=', caption_output)
            cur_word=caption_output.item()
            predictions.append(cur_word)
            self.print_debug('-------- next input------')
            if cur_word == 1:
                self.print_debug('equal to end')
                break
            
        return predictions

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#          super(LSTMTagger, self).__init__()

#         self.hidden_dim = hidden_dim

#         # embedding layer that turns words into a vector of a specified size
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

#         # the LSTM takes embedded word vectors (of a specified size) as inputs
#         # and outputs hidden states of size hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)

#         # the linear layer that maps the hidden state output dimension
#         # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

#         # initialize the hidden state (see code below)
#         self.hidden = self.init_hidden()


#     def forward(self, features, captions):
#         pass

#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         pass
