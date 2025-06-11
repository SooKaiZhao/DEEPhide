from docx import Document

class WhitespaceDecoder:
    """
    Decodes a message from a document encoded with trailing space steganography.
    Each line's trailing space encodes a bit: no extra space = 0, one extra space = 1.
    The first 32 bits are the message length.
    """
    @staticmethod
    def decode(encoded_doc_path, output_path):
        try:
            doc = Document(encoded_doc_path)
            binary_data = ''
            for paragraph in doc.paragraphs:
                text = paragraph.text
                if text.endswith(' '):
                    binary_data += '1'
                else:
                    binary_data += '0'
            if len(binary_data) < 32:
                return False, "Invalid encoded file: too short to contain length information"
            length_binary = binary_data[:32]
            message_length = int(length_binary, 2)
            message_binary = binary_data[32:32 + message_length * 8]
            if len(message_binary) < message_length * 8:
                return False, "Invalid encoded file: message data is incomplete"
            message = ''
            for i in range(0, len(message_binary), 8):
                byte = message_binary[i:i+8]
                if len(byte) == 8:
                    message += chr(int(byte, 2))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(message)
            return True, "Message decoded successfully!"
        except Exception as e:
            return False, f"Decoding error: {str(e)}"
