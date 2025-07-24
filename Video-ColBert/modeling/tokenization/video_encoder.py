import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class VideoEncoder(nn.Module):
    def __init__(self, 
                 visual_dim=1024,     # CLIP output size
                 audio_dim=128,       # VGGish output size
                 text_dim=768,        # BERT output size
                 fusion_dim=128,       # ColBERT embedding size
                 dropout=0.1):
        """
        Encodes video segments into ColBERT-compatible embeddings using early fusion
        Input: Pre-extracted features for a video segment (3-5 seconds)
        Output: Normalized 128-D embedding for late interaction
        """
        super().__init__()
        
        # Modality-specific normalization
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        
        # Text encoder (frozen BERT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.text_model.parameters():
            param.requires_grad = False  # Freeze BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + audio_dim + text_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, fusion_dim)
        )

    def forward(self, visual_feats, audio_feats, asr_text):
        """
        Args:
            visual_feats: [num_frames, visual_dim] from CLIP/ViT
            audio_feats: [num_windows, audio_dim] from VGGish/Wav2Vec
            asr_text: String of ASR transcript
        Returns:
            segment_embed: [1, fusion_dim] L2-normalized embedding
        """
        # 1. Modality-specific processing
        # Visual: Temporal average pooling
        visual_embed = self.visual_norm(visual_feats).mean(dim=0)  # [visual_dim]
        
        # Audio: Temporal average pooling
        audio_embed = self.audio_norm(audio_feats).mean(dim=0)     # [audio_dim]
        
        # Text: Mean pooling of BERT embeddings
        text_inputs = self.tokenizer(
            asr_text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32
        ).to(visual_feats.device)
        
        with torch.no_grad():
            text_output = self.text_model(**text_inputs)
        text_embed = text_output.last_hidden_state.mean(dim=1).squeeze()  # [text_dim]
        
        # 2. Early fusion
        combined = torch.cat([
            visual_embed, 
            audio_embed, 
            text_embed
        ], dim=-1)  # [visual_dim + audio_dim + text_dim]
        
        # 3. Project to ColBERT dimension
        segment_embed = self.fusion(combined)  # [fusion_dim]
        
        # 4. L2 normalization for cosine similarity
        return F.normalize(segment_embed, p=2, dim=-1)

    def det_embedding(self, visual_feats, audio_feats, asr_text):
        """
        Directly returns the embedding without normalization.
        Useful for retrieval tasks where raw embeddings are needed.
        """
        segment_embed = self.forward(visual_feats, audio_feats, asr_text)
        return segment_embed.detach()