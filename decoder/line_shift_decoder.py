from docx import Document

class LineShiftDecoder:
    @staticmethod
    def decode(encoded_doc_path, output_path):
        try:
            # Load the document
            doc = Document(encoded_doc_path)
            
            # Extract binary message
            binary_data = ''
            for paragraph in doc.paragraphs:
                if paragraph.paragraph_format.line_spacing == 1.15:
                    binary_data += '1'
                else:
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
