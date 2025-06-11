import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from text_preprocessor import preprocess_docx

from encoder.line_shift_encoder import LineShiftEncoder
from encoder.word_shift_encoder import WordShiftEncoder
from encoder.font_variation_encoder import FontVariationEncoder
from encoder.invisible_char_encoder import InvisibleCharEncoder
from encoder.whitespace_encoder import WhitespaceEncoder
from decoder.line_shift_decoder import LineShiftDecoder
from decoder.word_shift_decoder import WordShiftDecoder
from decoder.font_variation_decoder import FontVariationDecoder
from decoder.invisible_char_decoder import InvisibleCharDecoder
from decoder.whitespace_decoder import WhitespaceDecoder

import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))

# Model loading for prediction
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'deephide_cnnrnn.pth')

class HybridClassifier(torch.nn.Module):
    """
    MLP as feature projector before CNN and RNN.
    Architecture: Input feature vector -> MLP (projection) -> reshape to sequence -> CNN -> RNN -> output classifier.
    """
    def __init__(self, input_dim, mlp_proj_dim=64, cnn_filters=32, rnn_hidden=32, num_classes=6, dropout=0.3):
        super().__init__()
        # MLP feature projector
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, mlp_proj_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        # CNN and RNN layers
        self.cnn = torch.nn.Conv1d(1, cnn_filters, kernel_size=3, padding=1)
        self.rnn = torch.nn.GRU(cnn_filters, rnn_hidden, batch_first=True, bidirectional=True)
        # Output classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_hidden * 2 * mlp_proj_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, num_classes)
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

def load_deephide_model():
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    label2idx = checkpoint['label2idx']
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    num_classes = len(label2idx)
    model = HybridClassifier(input_dim=16, num_classes=num_classes)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, idx2label, label2idx

def predict_top3_methods(model, idx2label, file_path, max_seq_len=4096):
    with open(file_path, 'rb') as f:
        data = f.read()
    x = np.frombuffer(data, dtype=np.uint8)[:max_seq_len]
    if len(x) < max_seq_len:
        x = np.pad(x, (0, max_seq_len - len(x)), 'constant')
    x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        top3_idx = probs.argsort()[-3:][::-1]
        top3_methods = [(idx2label[i], float(probs[i])) for i in top3_idx]
    return top3_methods

class SteganographyTool:
    def __init__(self, root):
        # --- Window Properties ---
        root.title("DEEPhide Steganography Tool")
        root.geometry("820x650")
        root.minsize(800, 600)
        root.configure(bg="#e7f0fa")
        # Center window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        # --- Style ---
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background="#e7f0fa", borderwidth=0)
        style.configure('TNotebook.Tab', background="#e0e0e0", foreground="#222", padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', '#e9e9ee')])
        style.configure('TFrame', background="#e7f0fa")
        style.configure('TLabel', background="#e9e9ee", foreground="#222")
        style.configure('TButton', background="#e0e0e0", foreground="#222", padding=8, relief="flat")
        style.map('TButton', background=[('active', '#d0d0ff')])
        style.configure('Rounded.TButton', borderwidth=0, relief="flat", padding=8, background="#e0e0e0", foreground="#222")
        style.map('Rounded.TButton', background=[('active', '#d0d0ff')])
        style.configure('TEntry', fieldbackground="#f4f4f6", foreground="#222", padding=6)
        style.configure('TSeparator', background="#e0e0e0")

        # --- Message Area ---
        self.message_var = tk.StringVar(value="Welcome to DEEPhide!")
        self.message_type = tk.StringVar(value="info")  # info, success, error, warning
        self.message_banner = tk.Label(root, textvariable=self.message_var, anchor="w", padx=12, pady=8, font=("Arial", 11, "bold"), fg="#222", bg="#e7f0fa")
        self.message_banner.pack(fill="x", padx=0, pady=(0,2))
        self.update_message("Welcome to DEEPhide!", "info")

        # --- Instructions Tab ---
        self.instructions_frame = ttk.Frame(self.notebook, style='TFrame')
        self.setup_instructions_tab()

    def setup_instructions_tab(self):
        instructions = (
            "How to Use DEEPhide Steganography Tool\n\n"
            "1. Encoding a Message:\n"
            "   - Go to the 'Encode' tab.\n"
            "   - Select a Cover File (Word document).\n"
            "   - Select a Message File (text file with your secret message).\n"
            "   - Choose an Encoding Method.\n"
            "   - Click 'Encode' to generate the encoded file.\n\n"
            "2. Decoding a Message:\n"
            "   - Go to the 'Decode' tab.\n"
            "   - Select the Encoded File.\n"
            "   - Select the method used).\n"
            "   - Click 'Decode' to extract the message.\n\n"
            "3. Predict Encoding Method:\n"
            "   - Go to the 'Predict' tab.\n"
            "   - Select the Encoded File.\n"
            "   - Click 'Predict Encoding Method' to predict the method\n"
            "Tips:\n"
            "- The colored banner at the top will show important messages and feedback.\n"
            "- Currently, only .docx files are supported as cover file for encoding.\n"
        )
        label = tk.Label(self.instructions_frame, text="Instructions", font=("Arial", 14, "bold"), bg="#e7f0fa", anchor="w")
        label.pack(padx=20, pady=(20,10), anchor="w")
        text = tk.Text(self.instructions_frame, wrap="word", height=18, width=70, font=("Arial", 11), bg="#e7f0fa", relief="flat", borderwidth=0)
        text.insert("1.0", instructions)
        text.config(state="disabled")
        text.pack(padx=20, pady=10, fill="both", expand=True)

    def update_message(self, msg, msg_type="info"):
        self.message_var.set(msg)
        self.message_type.set(msg_type)
        colors = {
            "info":   {"bg": "#e7f0fa", "fg": "#222"},
            "success": {"bg": "#d4f8e8", "fg": "#225522"},
            "error":  {"bg": "#ffeaea", "fg": "#a00"},
            "warning": {"bg": "#fffacc", "fg": "#b48a00"},
        }
        c = colors.get(msg_type, colors["info"])
        self.message_banner.config(bg=c["bg"], fg=c["fg"])

        # --- Main Frames ---
        # Remove main_frame and pack notebook directly into root, filling all space above the status bar
        self.notebook = ttk.Notebook(root)
        self.encoder_frame = ttk.Frame(self.notebook, padding=20, style='TFrame')
        self.decoder_frame = ttk.Frame(self.notebook, padding=20, style='TFrame')
        self.predict_frame = ttk.Frame(self.notebook, padding=20, style='TFrame')
        self.instructions_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.encoder_frame, text="Encode")
        self.notebook.add(self.decoder_frame, text="Decode")
        self.notebook.add(self.predict_frame, text="Predict")
        self.notebook.add(self.instructions_frame, text="Instructions")
        self.notebook.pack(fill="both", expand=True, padx=6, pady=(6,0))
        self.setup_instructions_tab()

        # --- Section: Encode ---
        encode_header = tk.Label(self.encoder_frame, text="Encode Message", font=("Arial", 16, "bold"), bg="#e7f0fa", fg="#222")
        encode_header.pack(anchor="w", pady=(0, 10))
        self.setup_encoder()
        ttk.Separator(self.encoder_frame, orient="horizontal").pack(fill="x", pady=15)

        # --- Section: Decode ---
        decode_header = tk.Label(self.decoder_frame, text="Decode", font=("Arial", 16, "bold"), bg="#e7f0fa", fg="#222")
        decode_header.pack(anchor="w", pady=(0, 10))
        self.setup_decoder()
        ttk.Separator(self.decoder_frame, orient="horizontal").pack(fill="x", pady=15)

        # --- Section: Predict ---
        predict_header = tk.Label(self.predict_frame, text="Predict Encoding Method", font=("Arial", 16, "bold"), bg="#e7f0fa", fg="#222")
        predict_header.pack(anchor="w", pady=(0, 10))
        self.setup_predictor()
        ttk.Separator(self.predict_frame, orient="horizontal").pack(fill="x", pady=15)

        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Label(root, textvariable=self.status_var, anchor="w", relief="groove", padx=8, pady=4, bg="#e0e0e0", fg="#222")
        status_bar.pack(side="bottom", fill="x")

        # --- Tooltips (Simple) ---
        self.add_tooltips()

    def setup_predictor(self):
        ttk.Label(self.predict_frame, text="Encoded File:").pack(pady=5)
        self.predict_file_path = tk.StringVar()
        ttk.Entry(self.predict_frame, textvariable=self.predict_file_path, width=50).pack(pady=5)
        ttk.Button(self.predict_frame, text="Browse Encoded File", 
                  command=lambda: self.browse_file(self.predict_file_path, [("Word Documents", "*.docx"), ("Text Files", "*.txt")])).pack(pady=5)
        ttk.Button(self.predict_frame, text="Predict Encoding Method", command=self.predict_method).pack(pady=20)
        ttk.Label(self.predict_frame, text="Prediction Results:").pack(pady=5)
        self.predict_results = tk.Text(self.predict_frame, height=10, width=80)
        self.predict_results.pack(pady=5)

    def add_tooltips(self):
        try:
            import idlelib.tooltip as tooltip
            # Add tooltips for main controls
            tooltip.Hovertip(self.encoder_method_combo, "Select encoding method.")
            tooltip.Hovertip(self.decoder_menu, "Select decoding method.")
        except Exception:
            pass

    def setup_encoder(self):
        ttk.Label(self.encoder_frame, text="Cover File:").pack(pady=5)
        self.cover_path = tk.StringVar()
        ttk.Entry(self.encoder_frame, textvariable=self.cover_path, width=50).pack(pady=5)
        ttk.Button(self.encoder_frame, text="Browse Cover File", 
                  command=lambda: self.browse_file(self.cover_path, [("Word Documents", "*.docx"), ("Text Files", "*.txt")])).pack(pady=5)
        ttk.Label(self.encoder_frame, text="Message File:").pack(pady=5)
        self.message_path = tk.StringVar()
        ttk.Entry(self.encoder_frame, textvariable=self.message_path, width=50).pack(pady=5)
        ttk.Button(self.encoder_frame, text="Browse Message File",
                  command=lambda: self.browse_file(self.message_path, [("Text Files", "*.txt")])).pack(pady=5)
        ttk.Label(self.encoder_frame, text="Encoding Method:").pack(pady=5)
        self.encoding_method = tk.StringVar(value="Line Shift")
        self.encoder_method_combo = ttk.Combobox(self.encoder_frame, values=[
            "Line Shift", "Word Shift", "Font Variation", "Invisible Characters", "Whitespace"
        ], state="readonly")
        self.encoder_method_combo.pack(pady=5)
        ttk.Button(self.encoder_frame, text="Encode", command=self.encode).pack(pady=20)

    def setup_decoder(self):
        ttk.Label(self.decoder_frame, text="Encoded File:").pack(pady=5)
        self.encoded_path = tk.StringVar()
        ttk.Entry(self.decoder_frame, textvariable=self.encoded_path, width=50).pack(pady=5)
        ttk.Button(self.decoder_frame, text="Browse Encoded File", 
                  command=lambda: self.browse_file(self.encoded_path, [("Word Documents", "*.docx"), ("Text Files", "*.txt")])).pack(pady=5)
        ttk.Label(self.decoder_frame, text="Decoding Method:").pack(pady=5)
        self.decoding_method = tk.StringVar(value="Line Shift")
        decoder_options = ["Line Shift", "Word Shift", "Font Variation", "Invisible Characters", "Whitespace"]
        self.decoder_menu = ttk.OptionMenu(self.decoder_frame, self.decoding_method, decoder_options[0], *decoder_options)
        self.decoder_menu.pack(pady=5)
        ttk.Button(self.decoder_frame, text="Decode", command=self.decode).pack(pady=20)
        ttk.Label(self.decoder_frame, text="Decoded Output:").pack(pady=5)
        self.decode_results = tk.Text(self.decoder_frame, height=15, width=80)
        self.decode_results.pack(pady=5)
        self.decode_results.pack(pady=5)
        ttk.Button(self.decoder_frame, text="Decode", command=self.decode).pack(pady=5)

    def browse_file(self, path_var, file_types):
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            path_var.set(file_path)

    def encode(self):
        if not self.cover_path.get() or not self.message_path.get():
            messagebox.showerror("Error", "Please select both cover and message files")
            return
        try:
            # Preprocess cover file
            temp_cover_path = self.cover_path.get() + ".preprocessed.docx"
            preprocess_docx(self.cover_path.get(), temp_cover_path)
            # Do NOT show temp file location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".docx",
                filetypes=[("Word Documents", "*.docx")]
            )
            if save_path:
                try:
                    if self.encoder_method_combo.get() == "Line Shift":
                        success, message = LineShiftEncoder.encode(
                            temp_cover_path,
                            self.message_path.get(),
                            save_path
                        )
                    elif self.encoder_method_combo.get() == "Word Shift":
                        success, message = WordShiftEncoder.encode(
                            temp_cover_path,
                            self.message_path.get(),
                            save_path
                        )
                    elif self.encoder_method_combo.get() == "Font Variation":
                        success, message = FontVariationEncoder.encode(
                            temp_cover_path,
                            self.message_path.get(),
                            save_path
                        )
                    elif self.encoder_method_combo.get() == "Invisible Characters":
                        success, message = InvisibleCharEncoder.encode(
                            temp_cover_path,
                            self.message_path.get(),
                            save_path
                        )
                    elif self.encoder_method_combo.get() == "Whitespace":
                        success, message = WhitespaceEncoder.encode(
                            temp_cover_path,
                            self.message_path.get(),
                            save_path
                        )
                except Exception as enc_e:
                    messagebox.showerror("Error", f"Encoding failed: {str(enc_e)}")
                    return
                # Remove temp file only after encoding is fully complete
                if os.path.exists(temp_cover_path):
                    os.remove(temp_cover_path)
                if success:
                    messagebox.showinfo("Success", message)
                else:
                    messagebox.showerror("Error", message)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def decode(self):
        if not self.encoded_path.get():
            messagebox.showerror("Error", "Please select an encoded file")
            return
        try:
            decoder_map = {
                "Line Shift": LineShiftDecoder,
                "Word Shift": WordShiftDecoder,
                "Font Variation": FontVariationDecoder,
                "Invisible Characters": InvisibleCharDecoder,
                "Whitespace": WhitespaceDecoder
            }
            selected_method = self.decoding_method.get()
            if selected_method and selected_method != "Model Prediction" and selected_method in decoder_map:
                decoder = decoder_map[selected_method]
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text Files", "*.txt")],
                    title=f"Save decoded output for {selected_method}"
                )
                if not save_path:
                    self.decode_results.insert(tk.END, f"\nNo save path chosen.\n")
                    return
                try:
                    success, message = decoder.decode(self.encoded_path.get(), save_path)
                    if success:
                        self.decode_results.insert(tk.END, f"\n{selected_method} decoded successfully: {message}\nSaved to: {save_path}\n")
                    else:
                        self.decode_results.insert(tk.END, f"\n{selected_method} decode failed: {message}\n")
                except Exception as e:
                    self.decode_results.insert(tk.END, f"\n{selected_method} decode error: {str(e)}\n")
            else:
                # Predict top 3 encoding methods
                top3 = predict_top3_methods(self.model, self.idx2label, self.encoded_path.get())
                self.decode_results.delete(1.0, tk.END)
                best_method = top3[0][0] if top3 else "Unknown"
                self.decode_results.insert(tk.END, f"The most possible encoding method is {best_method}\n")
                # Try decoding with all top 3
                decoder_map2 = {
                    "LineShift": LineShiftDecoder,
                    "WordShift": WordShiftDecoder,
                    "FontVariation": FontVariationDecoder,
                    "InvisibleChar": InvisibleCharDecoder,
                    "Whitespace": WhitespaceDecoder,
                    "Line Shift": LineShiftDecoder,
                    "Word Shift": WordShiftDecoder,
                    "Font Variation": FontVariationDecoder,
                    "Invisible Characters": InvisibleCharDecoder,
                    "Whitespace": WhitespaceDecoder
                }
                for method, _ in top3:
                    decoder = decoder_map2.get(method, None)
                    if decoder is None:
                        self.decode_results.insert(tk.END, f"\nDecoder for {method} not found.\n")
                        continue
                    save_path = filedialog.asksaveasfilename(
                        defaultextension=".txt",
                        filetypes=[("Text Files", "*.txt")],
                        title=f"Save decoded output for {method}"
                    )
                    if not save_path:
                        self.decode_results.insert(tk.END, f"\nSkipping {method} (no save path chosen).\n")
                        continue
                    try:
                        success, message = decoder.decode(self.encoded_path.get(), save_path)
                        if success:
                            self.decode_results.insert(tk.END, f"\n{method} decoded successfully: {message}\nSaved to: {save_path}\n")
                        else:
                            self.decode_results.insert(tk.END, f"\n{method} decode failed: {message}\n")
                    except Exception as e:
                        self.decode_results.insert(tk.END, f"\n{method} decode error: {str(e)}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def predict_method(self):
        if not self.predict_file_path.get():
            messagebox.showerror("Error", "Please select an encoded file for prediction")
            return
        try:
            from dataset.predict_deephide_method import predict_encoding_method
            top3 = predict_encoding_method(self.predict_file_path.get())
            best_method = top3[0][0] if top3 else "Unknown"
            self.predict_results.delete(1.0, tk.END)
            self.predict_results.insert(tk.END, f"The most possible encoding method is: {best_method}\n")
            self.update_message(f"Prediction complete for {self.predict_file_path.get().split('/')[-1]}. Method: {best_method}", "success")
        except Exception as e:
            self.update_message(f"Prediction failed: {str(e)}", "error")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganographyTool(root)
    root.mainloop()
