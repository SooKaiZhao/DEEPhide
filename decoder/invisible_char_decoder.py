from docx import Document

class InvisibleCharDecoder:
    # Use zero-width non-joiner for encoding (for WPS Office compatibility)
    ZERO_WIDTH_NON_JOINER = '\u200C'  # Encodes bit '1', absence encodes '0'
    
    @staticmethod
    def decode(encoded_doc_path, output_path):
        try:
            doc = Document(encoded_doc_path)
            binary_data = ''
            for paragraph in doc.paragraphs:
                text = ''
                for run in paragraph.runs:
                    text += run.text
                i = 0
                while i < len(text):
                    char = text[i]
                    # Check for zero-width non-joiner after a character
                    if i+1 < len(text) and text[i+1] == InvisibleCharDecoder.ZERO_WIDTH_NON_JOINER:
                        binary_data += '1'
                        i += 2  # Skip both char and zero-width non-joiner
                    else:
                        binary_data += '0'
                        i += 1
            # First 32 bits contain the message length
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
