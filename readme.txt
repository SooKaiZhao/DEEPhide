DEEPhide: Deep Learning-Based Text Steganography Tool
====================================================

Overview
--------
DEEPhide is an advanced toolkit for text steganography, enabling users to hide and detect secret messages within text documents using both traditional and AI-powered methods. It supports multiple encoding techniques and leverages a deep learning model to predict the most likely steganographic method used in a given file.

Features
--------
- **Multiple Encoding Methods:**
  - Line Shift Coding
  - Word Shift Coding
  - Special Font Encoding
  - Invisible Character Encoding
  - Whitespace (Trailing Space) Steganography
- **User-Friendly GUI:** Built with Tkinter for easy operation.
- **Automatic Method Detection:** Predicts the encoding method using a CNN-RNN deep learning model.
- **Partial Decoding Verification:** Attempts partial decoding for every method to improve prediction reliability.
- **Method Prediction:** Shows the most probable encoding methods for any encoded file.
- **Extensible Design:** Modular structure allows for easy addition of new steganography techniques.

Installation
------------
1. Ensure you have Python 3.7 or later installed.
2. Install the required Python packages:

   pip install torch numpy pandas

   (You may also need: tkinter, depending on your Python distribution)

3. Place your trained model file (`deephide_cnnrnn.pth`) in the `dataset` directory.
4. Run the application:

   python stego_tool.py

Usage
-----
1. **Encoding:**
   - Go to the Encoder tab.
   - Select a cover file, choose an encoding method, and enter your secret message.
   - Save the encoded (stego) file.

2. **Decoding:**
   - Go to the Decoder tab.
   - Select an encoded file and the method used for encoding.
   - Extract and save the hidden message.

3. **Prediction:**
   - Unsure of the encoding method? Use the prediction feature to display the top 3 most likely methods, combining model confidence and partial decoding results.

Notes
-----
- The tool uses `torch.load` to load the model; only use trusted model files.
- For best results, use plain text or .docx files as input.
- The tool is designed for research, education, and secure communication purposes.

Contributing
------------
Contributions are welcome! You can add new encoders/decoders by following the modular class structure in the `encoder` and `decoder` directories.

License
-------
This project is provided for academic, research, and non-commercial use. See LICENSE file (if provided) for details.

Contact
-------
For questions or support, contact the project maintainer.
