import torch
from torch import nn
from utils import seq_max_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seq_and_vec(x):
    """
    seq shape: [None, seq_len, s_size]，
    vec shape: [None, v_size]
    Concat vec seq_len times: [None, seq_len, s_size+v_size]
    """
    seq, vec = x
    vec = torch.unsqueeze(vec, 1)

    vec = torch.zeros_like(seq[:, :, :1]) + vec
    return torch.cat([seq, vec], 2)


def seq_gather(x):
    """
    seq shape: [None, seq_len, s_size]
    idxs shape: [None,]
    Select vec from seq: [None, s_size]
    """
    seq, idxs = x
    batch_size = seq.shape[0]
    res = []
    for i in range(batch_size):
        vec = seq[i, idxs[i], :]
        res.append(vec)
    res = torch.stack(res, dim=0)
    return res


class DialatedGatedConv1d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding, dilation=1):
        super(DialatedGatedConv1d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.conv1 = nn.Conv1d(input_channel, output_channel,
                               kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(input_channel, output_channel,
                               kernel_size, padding=padding, dilation=dilation)
        if input_channel != output_channel:
            self.trans = nn.Conv1d(input_channel, output_channel, 1)

    def forward(self, args):
        X, attention_mask = args
        X = X * attention_mask
        gate = torch.sigmoid(self.conv2(X))
        if self.input_channel == self.output_channel:
            Y = X*(1-gate)+self.conv1(X)*gate
        else:
            Y = self.trans(X)*(1-gate)+self.conv1(X)*gate
        Y = Y*attention_mask
        return Y, attention_mask

    
class SubjectModel(nn.Module):
    def __init__(self, word_dict_length, word_emb_size):
        super(SubjectModel, self).__init__()

        self.embeds = nn.Embedding(word_dict_length, word_emb_size)
        self.fc1_dropout = nn.Sequential(
            nn.Dropout(0.25),  # drop 20% of the neuron
        )

        self.lstm1 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            input_size=word_emb_size,
            hidden_size=int(word_emb_size/2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # requires (batch, channel, seq)
        self.dgcnn = nn.Sequential(DialatedGatedConv1d(word_emb_size,word_emb_size,1,padding='same',dilation=1),
                   DialatedGatedConv1d(word_emb_size,word_emb_size,3,padding='same',dilation=1),
                   DialatedGatedConv1d(word_emb_size,word_emb_size,3,padding='same',dilation=2),
                   DialatedGatedConv1d(word_emb_size,word_emb_size,3,padding='same',dilation=4))

        # requires (batch, seq, channel)
        encoder_layer = nn.TransformerEncoderLayer(d_model=word_emb_size, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 2,  
                out_channels=word_emb_size,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size, 1),
        )

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size, 1),
        )

    def forward(self, tokens, attention_mask=None):
        """
        Performs forward and backward propagation and updates weights

        Parameters
        ----------
        tokens: tensor
            (batch_size, sent_len) a batch of tokenized texts
        attention_mask: tensor
            (batch_size, sent_len) attention mask for each text

        Returns
        -------
        subject_preds: tensor
            (batch_size, sent_len, 2)
        hidden_states: tensor
            (batch_size, sent_len, embed_size)
        """
        if attention_mask is None:
            attention_mask = torch.gt(tokens, 0).type(
                torch.FloatTensor).to(device)  # (batch_size,sent_len,1)
            attention_mask.requires_grad = False
        attention_mask = torch.unsqueeze(attention_mask, dim=2)

        outs = self.embeds(tokens) # (bsz, sent, emb)
        hidden_states = outs
        hidden_states = self.fc1_dropout(hidden_states)
        hidden_states = hidden_states.mul(attention_mask)  # (bsz, sent, emb)

        hidden_states, (h_n, c_n) = self.lstm1(hidden_states, None) # (bsz, sent, emb)
        hidden_states, (h_n, c_n) = self.lstm2(hidden_states, None) # (bsz, sent, emb)

        hidden_states = self.transformer_encoder(hidden_states) # (bsz, sent, emb)

        hidden_max, hidden_max_index = seq_max_pool([hidden_states, attention_mask]) # (bsz, emb)
        hidden_dim = list(hidden_states.size())[-1]
        h = seq_and_vec([hidden_states, hidden_max]) # (bsz, sent, emb * 2)

        h = h.permute(0, 2, 1)
        h = self.conv1(h)
        h = h.permute(0, 2, 1) # (bsz, sent, emb)

        ps1 = self.fc_ps1(h) # (bsz, sent, 1)
        ps2 = self.fc_ps2(h)

        subject_preds = torch.cat((ps1, ps2), dim=2)

        if attention_mask is not None:
            subject_preds *= attention_mask

        subject_preds = torch.sigmoid(subject_preds)

        return [subject_preds, hidden_states]


class CondLayerNorm(nn.Module):
    def __init__(self, embed_size, encoder_hidden=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_size, elementwise_affine=True)
        if encoder_hidden:
            self.gamma_encoder = nn.Sequential(
                nn.Linear(in_features=embed_size*2, out_features= encoder_hidden),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden, out_features=embed_size)
            )
            self.beta_encoder = nn.Sequential(
                nn.Linear(in_features=embed_size*2, out_features= encoder_hidden),
                nn.ReLU(),
                nn.Linear(in_features=encoder_hidden, out_features=embed_size)
            )
        else:
            self.gamma_encoder = nn.Linear(in_features=embed_size*2, out_features=embed_size) 
            self.beta_encoder = nn.Linear(in_features=embed_size*2, out_features=embed_size) 

    def forward(self, hidden_states, subject):
        """
        Perform layer normalization with conditions derived from subject embeddings
        
        Parameters
        ----------
        hidden_states: tensor
            (batch_size, sent_len, embed_size) hidden states generated from bert
        subject: tensor
            (batch_size, 2*embed_size) concatenation of the start and end of a sampled subject
            
        Returns
        -------
        normalized: tensor
            (batch_size, sent_len, embed_size) conditional-normalized hidden states
        """       
        std, mean = torch.std_mean(hidden_states, dim=-1, unbiased=False, keepdim=True)
        gamma = self.gamma_encoder(subject) # encoder output: (bsz, word_embed)
        beta = self.beta_encoder(subject)
        gamma = gamma.view(-1, 1, gamma.shape[-1]) # (bsz, 1, word_embed_size)
        beta = beta.view(-1, 1, beta.shape[-1]) # (bsz, 1, word_embed_size)
        normalized = (hidden_states - mean) / std * gamma + beta # hidden states: (bsz, sent_len, word_embed_size)
        return normalized

    
class ObjectModel(nn.Module):
    def __init__(self, word_emb_size, num_classes):
        super(ObjectModel, self).__init__()

        self.cond_layer_norm = CondLayerNorm(word_emb_size, encoder_hidden=word_emb_size//2)

        self.fc_ps1 = nn.Sequential(
            nn.Linear(word_emb_size, num_classes),
            # nn.Sigmoid(),
        )

        self.fc_ps2 = nn.Sequential(
            nn.Linear(word_emb_size, num_classes),
            # nn.Sigmoid(),
        )

    def forward(self, hidden_states, suject_pos, attention_mask=None):
        """
        Extract objects with given subject positions
        
        Parameters
        ----------
        hidden_states: tensor
            (batch_size, sent_len, embed_size) hidden states generated from bert
        subject_pos: tensor
            (batch_size, 2) start and end position of a sampled subject
        attention_mask: tensor
            (batch_size, sent_len) attention mask for each text

        Returns
        -------
        preds: tensor
            (batch_size, sent_len, predicate_num, 2) conditional-normalized hidden states
        """ 

        subj_head = suject_pos[:, 0] # (bsz)
        subj_tail = suject_pos[:, 1]
        if attention_mask is not None:
            # (bsz, emb)
            hidden_max, _ = seq_max_pool([hidden_states, attention_mask.unsqueeze(dim=2)])
        else:
            hidden_max, _ = seq_max_pool([hidden_states, torch.ones((hidden_states.shape[0], hidden_states.shape[1], 1))])

        # extract subject head and tail's embedding
        subj_head_emb = seq_gather([hidden_states, subj_head]) # (bsz, emb)
        subj_tail_emb = seq_gather([hidden_states, subj_tail])
        subj_emb = torch.cat([subj_head_emb, subj_tail_emb], 1) # (bsz, emb*2)
        
        h = self.cond_layer_norm(hidden_states, subj_emb) # (bsz, sent_len, emb_size)

        po1 = self.fc_ps1(h) # (bsz, sent, cls)
        po2 = self.fc_ps2(h)

        if attention_mask is not None:
            po1 *= attention_mask.unsqueeze(dim=2)
            po2 *= attention_mask.unsqueeze(dim=2)

        po1 = torch.sigmoid(po1)
        po2 = torch.sigmoid(po2)

        object_preds = torch.stack((po1, po2), dim=3)

        return object_preds
