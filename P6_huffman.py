import cv2
import numpy as np
import heapq
import argparse
from collections import defaultdict
from math import log2

# Define a class for the nodes in the Huffman tree
class HuffmanTreeNode:
    def __init__(self, pixel_value, frequency):
        self.pixel_value = pixel_value
        self.frequency = frequency
        self.left = None
        self.right = None

    # Define comparison operation for the priority queue (heap)
    def __lt__(self, other):
        return self.frequency < other.frequency

# Function to calculate the histogram and probabilities of pixel intensities
def calculateHistogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = np.sum(histogram)
    probabilities = histogram / total_pixels
    return probabilities

# Function to construct the Huffman tree from the pixel probabilities
def constructHuffmanTree(probabilities):
    priority_queue = [HuffmanTreeNode(value, prob) for value, prob in enumerate(probabilities) if prob > 0]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanTreeNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

# Function to recursively generate Huffman codes from the tree
def generateHuffmanCodes(node, current_code="", codebook=None):
    if codebook is None:
        codebook = {}

    if node.pixel_value is not None:
        codebook[node.pixel_value] = current_code
    else:
        generateHuffmanCodes(node.left, current_code + "0", codebook)
        generateHuffmanCodes(node.right, current_code + "1", codebook)

    return codebook

# Function to compute entropy (theoretical minimum average code length)
def computeEntropy(probabilities):
    return -np.sum([p * log2(p) for p in probabilities if p > 0])

# Function to calculate the compression ratio based on Huffman coding
def calculateCompressionRatio(probabilities, codebook):
    avg_code_length = np.sum([probabilities[symbol] * len(code) for symbol, code in codebook.items()])
    return 8 / avg_code_length  # Assuming 8 bits per pixel

# Function to compute Shannon's theoretical limit for code length
def computeShannonLimit(probabilities):
    return -np.sum([p * log2(p) for p in probabilities if p > 0])

# Function to display the Huffman codebook
def displayHuffmanCodes(codebook):
    for symbol, code in codebook.items():
        print(f"Pixel Value: {symbol}, Huffman Code: {code}")

# Main function to run the program
def main():
    parser = argparse.ArgumentParser(description="Compute Huffman codes, entropy, and compression ratio for a grayscale image.")
    parser.add_argument("imagefile", help="Path to the input grayscale image.")
    args = parser.parse_args()

    # Load the grayscale image
    image = cv2.imread(args.imagefile, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Calculate the histogram and probabilities
    probabilities = calculateHistogram(image)

    # Construct the Huffman tree and generate codes
    huffman_tree = constructHuffmanTree(probabilities)
    huffman_codes = generateHuffmanCodes(huffman_tree)

    # Compute the entropy of the image
    entropy = computeEntropy(probabilities)

    # Compute the compression ratio achieved by Huffman coding
    compression_ratio = calculateCompressionRatio(probabilities, huffman_codes)

    # Compute Shannon's theoretical limit for the average code length
    shannon_limit_value = computeShannonLimit(probabilities)

    # Display Huffman codes
    print("Huffman Codebook:")
    displayHuffmanCodes(huffman_codes)

    # Display calculated values
    print(f"\nEntropy: {entropy:.4f} bits/pixel")
    print(f"Compression Ratio: {compression_ratio:.4f}")
    print(f"Shannon's Theoretical Limit: {shannon_limit_value:.4f} bits/pixel")

if __name__ == "__main__":
    main()
