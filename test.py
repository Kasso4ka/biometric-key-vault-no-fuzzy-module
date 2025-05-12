import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import requests
import base64

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"


def visualize_binary_codes(binary_codes, title="Binary Codes", save_path=None):
    if not len(binary_codes):
        logger.info("No binary codes to visualize")
        return

    num_codes = len(binary_codes)
    code_length = len(binary_codes[0])

    plt.figure(figsize=(12, 2))
    plt.imshow(binary_codes, cmap='binary', aspect='auto')
    plt.colorbar(ticks=[0, 1], label='Bit Value')
    plt.xlabel('Bit Position')
    plt.ylabel('Face Index')
    plt.title(f"{title} - {num_codes} faces")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Visualization saved to {save_path}")

    plt.show()


def extract_frames_from_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return []

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video")
    return frames


def encode_image_to_base64(image):
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return None
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health_info = response.json()
            logger.info(f"API is healthy: {health_info}")
            return True
        else:
            logger.error(
                f"API health check failed with status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Failed to connect to API: {str(e)}")
        return False


def test_video_via_api():
    if not check_api_health():
        logger.error("API is not available. Make sure the server is running.")
        return None

    video_path = "./videos/lexa.mp4"
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    logger.info(f"Processing video via API: {video_path}")
    start_time = time.time()

    with open(video_path, 'rb') as video_file:
        files = {'video': (os.path.basename(video_path),
                           video_file, 'video/mp4')}

        try:
            response = requests.post(f"{API_URL}/process_video", files=files)

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status code {response.status_code}: {response.text}")
                return None

            result = response.json()
            binary_codes = result.get('binary_codes', [])
            api_processing_time = result.get('processing_time', 0)
            settled_vector = result.get('settled_vector', [])
            mnemonic = result.get('mnemonic', "")
            total_time = time.time() - start_time
            logger.info(f"API processing time: {api_processing_time:.2f}s")
            logger.info(f"Total request time: {total_time:.2f}s")
            logger.info(f"Found {len(binary_codes)} faces in video")
            logger.info(f"Settled vector: {settled_vector}")
            logger.info(f"Mnemonic: {mnemonic}")

            if len(binary_codes):
                binary_array = np.array(binary_codes)
                logger.info(f"Binary codes shape: {binary_array.shape}")

                visualize_binary_codes(binary_array,
                                       title="Binary Codes from Video (via API)",
                                       save_path="video_binary_codes_api.png")
            else:
                logger.warning("No binary codes were generated")

            return binary_codes

        except Exception as e:
            logger.error(f"Error sending request to API: {str(e)}")
            return None


def test_frames_via_api():
    if not check_api_health():
        logger.error("API is not available. Make sure the server is running.")
        return None

    video_path = "./videos/lexa.mp4"
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    logger.info(f"Extracting frames from video: {video_path}")
    frames = extract_frames_from_video(video_path)

    if not len(frames):
        logger.error("No frames extracted from video")
        return None

    logger.info(f"Encoding {len(frames)} frames to base64")
    encoded_frames = []
    for frame in frames:
        encoded_frame = encode_image_to_base64(frame)
        if encoded_frame:
            encoded_frames.append(encoded_frame)

    if not len(encoded_frames):
        logger.error("Failed to encode frames")
        return None

    logger.info(f"Processing {len(encoded_frames)} frames via API")
    start_time = time.time()

    try:
        payload = {"images": encoded_frames}
        response = requests.post(f"{API_URL}/process_images", json=payload)

        if response.status_code != 200:
            logger.error(
                f"API request failed with status code {response.status_code}: {response.text}")
            return None

        result = response.json()
        binary_codes = result.get('binary_codes', [])
        api_processing_time = result.get('processing_time', 0)
        settled_vector = result.get('settled_vector', [])
        mnemonic = result.get('mnemonic', "")
        total_time = time.time() - start_time
        logger.info(f"API processing time: {api_processing_time:.2f}s")
        logger.info(f"Total request time: {total_time:.2f}s")
        logger.info(f"Found {len(binary_codes)} faces in frames")
        logger.info(f"Settled vector: {settled_vector}")
        logger.info(f"Mnemonic: {mnemonic}")

        if len(binary_codes):
            binary_array = np.array(binary_codes)
            logger.info(f"Binary codes shape: {binary_array.shape}")

            visualize_binary_codes(binary_array,
                                   title="Binary Codes from Frames (via API)",
                                   save_path="frames_binary_codes_api.png")
        else:
            logger.warning("No binary codes were generated")

        return binary_codes

    except Exception as e:
        logger.error(f"Error sending request to API: {str(e)}")
        return None


def compare_results(video_codes, frames_codes):
    if video_codes is None or frames_codes is None:
        logger.error("Cannot compare: one or both results are None")
        return

    if not len(video_codes) or not len(frames_codes):
        logger.error("Cannot compare: one or both results are empty")
        return

    logger.info(
        f"Comparing results: {len(video_codes)} codes from video, {len(frames_codes)} codes from frames")

    video_array = np.array(video_codes)
    frames_array = np.array(frames_codes)

    min_len = min(len(video_array), len(frames_array))
    if min_len == 0:
        logger.error("Cannot compare: no common elements")
        return

    if len(video_array) != len(frames_array):
        logger.info(
            f"Note: Different number of faces detected: {len(video_array)} in video vs {len(frames_array)} in frames")
        logger.info(f"Comparing only the first {min_len} faces")
        video_array = video_array[:min_len]
        frames_array = frames_array[:min_len]

    matches = np.all(video_array == frames_array, axis=1).sum()
    match_percentage = matches / min_len * 100
    logger.info(
        f"Matching binary codes: {matches}/{min_len} ({match_percentage:.2f}%)")

    if match_percentage < 100:
        hamming_distances = np.sum(video_array != frames_array, axis=1)
        avg_hamming = np.mean(hamming_distances)
        logger.info(
            f"Average Hamming distance for non-matching codes: {avg_hamming:.2f} bits")

        plt.figure(figsize=(10, 4))
        plt.hist(hamming_distances, bins=range(17), alpha=0.7)
        plt.xlabel("Hamming Distance (bits)")
        plt.ylabel("Count")
        plt.title(
            "Hamming Distance Distribution between Video and Frame Binary Codes")
        plt.grid(alpha=0.3)
        plt.savefig("hamming_distances.png", dpi=150)
        plt.show()

        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        ax[0].imshow(video_array, cmap='binary', aspect='auto')
        ax[0].set_title("Binary Codes from Video API")
        ax[0].set_ylabel("Face Index")

        ax[1].imshow(frames_array, cmap='binary', aspect='auto')
        ax[1].set_title("Binary Codes from Frames API")
        ax[1].set_ylabel("Face Index")
        ax[1].set_xlabel("Bit Position")

        plt.tight_layout()
        plt.savefig("comparison.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    # Run both tests
    logger.info("==== Testing with video API endpoint ====")
    video_binary_codes = test_video_via_api()

    logger.info("\n==== Testing with frames/images API endpoint ====")
    frames_binary_codes = test_frames_via_api()

    # Compare results
    logger.info("\n==== Comparing results ====")
    compare_results(video_binary_codes, frames_binary_codes)
