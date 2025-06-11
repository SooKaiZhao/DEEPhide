import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

TRAINSET_DIR = r"C:\Users\User\Desktop\DEEPhide\TrainingSet"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deephide_cnnrnn.pth')
MAX_SEQ_LEN = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WHITESPACE_LABEL = "Whitespace"

class HybridClassifier(nn.Module):
    """
    MLP as feature projector before CNN and RNN.
    Architecture: Input feature vector -> MLP (projection) -> reshape to sequence -> CNN -> RNN -> output classifier.
    """
    def __init__(self, input_dim, mlp_proj_dim=64, cnn_filters=32, rnn_hidden=32, num_classes=6, dropout=0.3):
        super().__init__()
        # MLP feature projector
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # CNN and RNN layers
        self.cnn = nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1)
        self.rnn = nn.GRU(cnn_filters, rnn_hidden, batch_first=True, bidirectional=True)
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2 * mlp_proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.mlp(x)  # [batch, mlp_proj_dim]
        x = x.unsqueeze(1)  # [batch, 1, mlp_proj_dim]
        x = self.cnn(x)     # [batch, cnn_filters, mlp_proj_dim]
        x = x.transpose(1, 2)  # [batch, mlp_proj_dim, cnn_filters]
        out, _ = self.rnn(x)   # [batch, mlp_proj_dim, 2*rnn_hidden]
        out = out.reshape(out.size(0), -1)  # Flatten all timesteps
        return self.classifier(out)

from feature_extractor import FeatureExtractor

def predict_encoding_method(encoded_file_path):
    # Extract features from the docx file
    features = FeatureExtractor.extract_features(encoded_file_path)
    # Convert feature dict to ordered vector (ensure consistent order)
    feature_keys = [
        'lines_with_trailing_space', 'max_trailing_space', 'min_trailing_space', 'mean_trailing_space', 'ratio_lines_with_trailing_space',
        'zero_width_char_count', 'ratio_zero_width_char', 'lines_with_zero_width_char',
        'paragraphs_with_non_default_spacing', 'ratio_paragraphs_non_default_spacing', 'unique_line_spacings',
        'double_space_count', 'ratio_double_space', 'paragraphs_with_double_space',
        'font_color_variation_count', 'ratio_colored_chars'
    ]
    feature_vector = [features.get(k, 0) for k in feature_keys]
    feature_vector = np.array(feature_vector, dtype=np.float32)
    x = torch.tensor(feature_vector).unsqueeze(0).to(DEVICE)

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    label2idx = checkpoint['label2idx']
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    num_classes = len(label2idx)
    model = HybridClassifier(input_dim=16, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top3_idx = probs.argsort()[-3:][::-1]
        top3_methods = [(idx2label[i], float(probs[i])) for i in top3_idx]
    return top3_methods


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Predict DEEPhide encoding method for a docx file')
    parser.add_argument('file', type=str, help='Path to encoded .docx file')
    args = parser.parse_args()
    top3 = predict_encoding_method(args.file)
    print("Top 3 predicted encoding methods:")
    for method, prob in top3:
        print(f"{method}: {prob:.3f}")
