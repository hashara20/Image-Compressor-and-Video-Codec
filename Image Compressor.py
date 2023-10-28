import numpy as np
from collections import Counter
import heapq
import cv2
from google.colab.patches import cv2_imshow


# read input image as grayscale
img = cv2.imread('tree.png', 0)

# convert the grayscale to float32
matrix = np.float32(img) # float conversion

height , width = matrix.shape

#size of the image
vertical =height//8
horizontal=width//8

#store 8*8 macroblocks in a List
macro_blocks = []

for i in range(horizontal):
    for j in range(vertical):

        block=matrix[j*8:j*8+8,i*8:i*8+8]
        macro_blocks.append(block)

# Aply Discrete cosine transform for each block
dct_blocks = []

for i in range(len(macro_blocks)):
    dct = cv2.dct(macro_blocks[i], cv2.DCT_INVERSE)
    dct_blocks.append((dct.round()).astype(int))



#apply quantization for each block
quantized_blocks = []
quantization_level=30  #quantizaation level

for i in range(len(macro_blocks)):
    quantized_blocks.append((np.round(dct_blocks[i] / quantization_level)).astype(int))

#flatten the list
flattened_list = []

for i in range(len(macro_blocks)):
    for j in range(8):
        for k in range(8):
            flattened_list.append(quantized_blocks[i][j][k])

# Huffman Encoding
class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

# Calculate frequency of each value
def calculate_frequency(values):
    frequency = Counter(values)
    return frequency

# Generate Huffman tree
def construct_huffman_tree(frequency):
    heap = []
    for value, freq in frequency.items():
        heapq.heappush(heap, Node(value, freq))

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0]

# Generate Huffman codebook
def generate_huffman_codes(root):
    codebook = {}

    def traverse(node, code):
        if node.value is not None:
            codebook[node.value] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codebook

# Encode values using Huffman codes
def encode_values(values, codebook):
    encoded_values = ''
    for value in values:
        encoded_values += codebook[value]
    return encoded_values



# Calculate frequency and construct Huffman tree
frequency = calculate_frequency(flattened_list)
huffman_tree = construct_huffman_tree(frequency)

# Generate Huffman codebook
codebook = generate_huffman_codes(huffman_tree)

# Encode values
encoded_values = encode_values(flattened_list, codebook)

# Pad the encoded data
padding_length = 8 - len(encoded_values) % 8
encoded_values += "0" * padding_length
padding_info = format(padding_length, '08b')
encoded_values = padding_info + encoded_values

# Convert encoded data to bytes
byte_array = bytearray()
for i in range(0, len(encoded_values), 8):
    byte = encoded_values[i:i + 8]
    byte_array.append(int(byte, 2))

# Save encoded data as a binary file
output_file = 'huffman_encoded_values.txt'
with open(output_file, 'wb') as file:
    file.write(byte_array)

print("Encoded data saved to 'huffman_encoded_values.txt' file.")

# Huffman Decoding
def decode_values(encoded_values, codebook):
    decoded_values = []
    current_code = ""
    for bit in encoded_values:
        current_code += bit
        for value, code in codebook.items():
            if current_code == code:
                decoded_values.append(value)
                current_code = ""
                break
    return decoded_values

# Read encoded values from the binary file
input_file = 'huffman_encoded_values.txt'
with open(input_file, 'rb') as file:
    byte_array = file.read()

# Convert byte array to binary string
binary_data = ""
for byte in byte_array:
    binary_data += format(byte, '08b')

padding_length = int(binary_data[:8], 2)
binary_data = binary_data[8:-padding_length]

decoded_values = decode_values(binary_data, codebook)

#re generate the list
regenerated_quantized = np.empty((vertical, horizontal), dtype=object)

# re form the 2D array as 8*8 blocks of elements
x=0
while(x<len(decoded_values)):
    for i in range(horizontal):
        for j in range(vertical):
            reqaunt = np.zeros((8, 8)).astype(int)
            for k in range(8):
                for l in range(8):
                    reqaunt[k, l] = decoded_values[x]
                    x+=1
                    regenerated_quantized[j,i]=reqaunt



# decontization
dequantized_blocks= np.empty((vertical, horizontal), dtype=object)

for i in range(horizontal):
    for j in range(vertical):
        dequantized_blocks[j,i]=(regenerated_quantized[j,i]*quantization_level).astype(float)


# apply inverse discrete cosine transform
inv_dct_blocks= np.empty((vertical, horizontal), dtype=object)

for i in range(horizontal):
    for j in range(vertical):
        inv_dct_blocks[j,i]= cv2.idct(dequantized_blocks[j,i])



# convert re generated 2D array into the image_matrix

def recontruct_img_matrix(idctblocks):

    imgmatrix = np.zeros((horizontal * 8, vertical * 8))

    for i in range(horizontal):
        for j in range(vertical):
            imgmatrix[i*8:i*8+8,j*8:j*8+8]=idctblocks[j,i].T

    return imgmatrix.T



#reconstructing the image
trimg = np.uint8(recontruct_img_matrix(inv_dct_blocks))

# display the images
cv2_imshow(img) # original image
cv2_imshow( trimg)  # transmit image
cv2.waitKey(0)
cv2.destroyAllWindows()
