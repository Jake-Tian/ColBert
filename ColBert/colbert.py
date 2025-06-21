import string
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            # Create a skiplist of punctuation symbols and their token IDs
            # {'symbol': True, 'token_id': True}
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(*Q), self.doc(*D))

    """ GPU: Model computations (BERT, linear layers, and normalization)
        CPU: Masking, list conversion, and other operations
    """
    def query(self, input_ids, attention_mask):
        """ Query Encoder: Process the query input and return normalized embeddings.
            Eq = Normalize(Linear(BERT([Q]q0q1...ql##...#))) where # refers to the [mask] token."""
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        # BERT will produce query-based embeddings for the [Q] token and the [mask] tokens.
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        """ Document Encoder: Process the document input and return normalized embeddings.
            Ed = Filter(Normalize(Linear(BERT([D]d0d1...dn))))"""
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        """Compute the similarity score between query and document embeddings.
        Cosine similarity (default): S_q,d = sum_i(max_j(Q_i * D_j))
        L2 similarity: S_q,d = sum_i(max_j(-||Q_i - D_j||^2))"""
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        """Create a mask for the input IDs, excluding tokens in the skiplist and padding tokens."""
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
