from docx import Document

class WordShiftDecoder:
    @staticmethod
    def decode(encoded_doc_path, output_path):
        try:
            # Load the document
            doc = Document(encoded_doc_path)
            
            # Extract binary message by analyzing word spacing
            binary_data = ''
            
            for paragraph in doc.paragraphs:
                # Look at each run in the paragraph
                for run in paragraph.runs:
                    text = run.text
                    # Check if the run ends with double space (right shift - '1')
                    if text.endswith('  '):
                        binary_data += '1'
                    # Check if the run ends with single space (left shift - '0')
                    elif text.endswith(' '):
                        binary_data += '0'
            
            # First 32 bits contain the message length
            if len(binary_data) < 32:
                return False, "Invalid encoded file: too short to contain length information"
                
            length_binary = binary_data[:32]
            message_length = int(length_binary, 2)
            
            # Extract message binary data
            message_binary = binary_data[32:32 + message_length * 8]
            
            # Convert binary to text
            message = ''
            for i in range(0, len(message_binary), 8):
                byte = message_binary[i:i+8]
                if len(byte) == 8:  # Ensure we have a complete byte
                    message += chr(int(byte, 2))
            
            # Save the decoded message
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(message)
            
            return True, "Message decoded successfully!"
            
        except Exception as e:
            return False, f"Decoding error: {str(e)}"
