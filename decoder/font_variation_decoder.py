from docx import Document
from docx.shared import RGBColor

class FontVariationDecoder:
    @staticmethod
    def decode(encoded_doc_path, output_path):
        try:
            # Load the document
            doc = Document(encoded_doc_path)
            
            # Extract binary message by analyzing font variations
            binary_data = ''
            
            for paragraph in doc.paragraphs:
                for run in paragraph.runs:
                    if run.text.isspace():
                        continue
                        
                    for char in run.text:
                        if char.isspace():
                            continue
                            
                        # Check color variation
                        if not hasattr(run.font.color, 'rgb') or run.font.color.rgb is None:
                            continue
                        
                        try:
                            # Get the RGB color value
                            rgb_str = str(run.font.color.rgb)
                            if not rgb_str:
                                continue
                                
                            # Remove any non-hex characters and ensure we have a valid hex color
                            rgb_hex = rgb_str.strip().replace('0x', '')
                            if len(rgb_hex) != 6:  # Should be 6 characters for RGB
                                continue
                            
                            # Convert hex to RGB values
                            r = int(rgb_hex[0:2], 16)
                            g = int(rgb_hex[2:4], 16)
                            b = int(rgb_hex[4:6], 16)
                            
                            # Check the color values
                            if r == 1 and g == 1 and b == 1:  # Almost black
                                binary_data += '1'
                            elif r == 2 and g == 2 and b == 2:  # Very slightly lighter
                                binary_data += '0'
                        except (ValueError, IndexError):
                            # Skip any invalid color values
                            continue
            
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
