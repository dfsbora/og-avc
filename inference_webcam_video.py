"""
Audio-Visual Inference Script

This script modifies the script inference_webcam_images.py to support inference
using video recorded from the laptop's webcam and microphone.

Usage:
- $ python inference_webcam_video.py  --experiment_name test_run1 [--save_visualizations]
[--model_dir checkpoints] [--recording_duration 10.0] [--segment_duration 1.5] [--data_dir data_video]

Notes:
- Recorded video saved at og_avc/data/video
- Processed video saved at og_avc/data/processed_video
- If saving individual frame visualizations, saved at og_avc/checkpoints/run_name/viz_seed_10
"""


import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from model_cnn import EZVSL
from datasets import get_inference_dataset, inverse_normalize
import cv2
import random
from torchvision.models import resnet18
import moviepy.editor as mp
from torchvision import transforms
from PIL import Image

import pyaudio
import wave
import threading
import time



def record_audio(filename, duration):
    """
    Record audio from the default microphone input and save it to a WAV file.

    Parameters:
    - filename (str): The name of the output WAV file.
    - duration (float): The duration of the recording in seconds.
    """

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("* Recording audio...")

    frames = []

    # Record in chunks
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write wav file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def record_video(filename, duration):
    """
    Record video from the default webcam.

    Parameters:
    - filename (str): The name of the output video.
    - duration (float): The duration of the recording in seconds.
    """
    cap = cv2.VideoCapture(0)

    # OpenCV video recording parameters
    fps = 30
    resolution = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # OpenCV VideoWriter object
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    # Start recording
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    # Release resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def capture_image(filename):
    """
    Capture an image from the default webcam and save it as an image file.

    Parameters:
    - filename (str): The name of the output image file.
    """
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cv2.imwrite(filename, frame)

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()


def merge_audio_video(video_file, audio_file, output_file):
    """
    Merges an audio file with a video file and saves the result as a new video file.
    """
    video_clip = mp.VideoFileClip(video_file)
    audio_clip = mp.AudioFileClip(audio_file)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
    print(f"Final video saved: {output_file}")


def record_video_with_audio(duration, data_dir):
    """
    Records a video with audio for a specified duration and saves the final output.

    Parameters:
    duration (int): Duration of the recording in seconds.
    data_dir (str): Directory where the temporary and final files will be stored.

    Returns:
    str: Path to the final merged video file.
    """

    # Directories and filenames
    temp_dir = os.path.join(data_dir, "temp")
    output_dir = os.path.join(data_dir, "video")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_filename = os.path.join(temp_dir, f"{timestamp}.avi")
    audio_filename = os.path.join(temp_dir, f"{timestamp}.wav")
    output_filename = os.path.join(output_dir, f"{timestamp}.mp4")

    # Create threads for recording audio and capturing image
    audio_thread = threading.Thread(target=record_audio, args=(audio_filename, duration))
    video_thread = threading.Thread(target=record_video, args=(video_filename, duration))

    # Start both threads
    audio_thread.start()
    video_thread.start()

    # Wait for both threads to finish
    audio_thread.join()
    video_thread.join()

    print(f"Temp audio and video saved: {audio_filename}, {video_filename}")

    merge_audio_video(video_filename, audio_filename, output_filename)

    os.remove(video_filename)

    return output_filename


def delete_temp_folder(temp_dir):
    """
    Deletes a temporary directory and all its contents.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    else:
        print(f"The folder '{temp_dir}' does not exist.")


class PredictionCanvas:
    def __init__(self, width, height):
        """
        Initializes the Canvas class with given images, dimensions, and labels information.

        Parameters:
        - width: Width of the canvas
        - height: Height of the canvas
        """
        self.width = width
        self.height = height
        self.canvas = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        self.img_pred_avc = np.zeros((width // 2, height // 2, 3), dtype=np.uint8)
        self.img_pred_og = np.zeros((width // 2, height // 2, 3), dtype=np.uint8)
        self.img_pred_detr = np.zeros((width // 2, height // 2, 3), dtype=np.uint8)
        self.img_pred_og_avc_detr = np.zeros((width // 2, height // 2, 3), dtype=np.uint8)
        self.labels_info = [
            ("AVC Prediction", (10, 30)),
            ("OG Prediction", (int(height / 2) + 10, 30)),
            ("DETR Prediction", (10, int(width / 2) + 30)),
            ("OG + AVC + DETR", (int(height / 2) + 10, int(width / 2) + 30))
        ]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.color = (255, 255, 255)
        self.thickness = 1

    def update_predictions(self, img_pred_avc, img_pred_og, img_pred_detr, img_pred_og_avc_detr):
        """
        Sets new prediction images.
        """
        self.img_pred_avc = img_pred_avc
        self.img_pred_og = img_pred_og
        self.img_pred_detr = img_pred_detr
        self.img_pred_og_avc_detr = img_pred_og_avc_detr

    def create_canvas(self):
        """
        Creates a canvas and arranges the prediction images in a grid with labels.
        """
        self.canvas[:int(self.width / 2), :int(self.height / 2)] = self.img_pred_avc
        self.canvas[:int(self.width / 2), int(self.height / 2):] = self.img_pred_og
        self.canvas[int(self.width / 2):, :int(self.height / 2)] = cv2.resize(self.img_pred_detr, (int(self.width / 2), int(self.height / 2)))
        self.canvas[int(self.width / 2):, int(self.height / 2):] = self.img_pred_og_avc_detr

        for label, position in self.labels_info:
            cv2.putText(self.canvas, label, position, self.font, self.font_scale, self.color, self.thickness)

    def get_canvas(self):
        return self.canvas


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='vggsound_144k_run1', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_visualizations', action='store_false', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')
    parser.add_argument('--evaluate_metrics', action='store_true', help='Set to evaluate the metrics using ground truth. Default not run')

    # Dataset
    #parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    #parser.add_argument('--test_data_path', default='/data2/dataset/labeled_5k_flicker/Data/', type=str, help='Root directory path of data')
    #parser.add_argument('--test_gt_path', default='/data2/dataset/labeled_5k_flicker/Annotations/', type=str)
    parser.add_argument('--testset', default='vggss', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='/data2/dataset/vggss/vggss_dataset_different_naming/', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='/data2/dataset/vggss/', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    # Distributed params
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument("--seed", type=list, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], help="random seed")

    # Recording parameters
    parser.add_argument('--recording_duration', default=15.0, type=float, help="Duration of audio recording in seconds")
    parser.add_argument('--segment_duration', default=1.5, type=float, help="Duration of processing segments in seconds")
    parser.add_argument('--data_dir', default='data', type=str, help="Data folder to save videos")

    return parser.parse_args()


def main(args):
    # Set up seed for reproducibility (up-sampling operation however is not reproducible)
    seed_num = args.seed[0]
    print(">>>>>> Testing seed for this round of test is {}.".format(seed_num))
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True

    # Set CUDA workspace configuration for deterministic behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Initialize model directory and visualization directory
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, "viz_seed_" + str(seed_num))
    os.makedirs(viz_dir, exist_ok=True)

    # Load models
    audio_visual_model = EZVSL(args.tau, args.out_dim)
    object_saliency_model = resnet18(weights="ResNet18_Weights.DEFAULT")
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )
    detr_model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

    # Set models to GPU
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            audio_visual_model.cuda(args.gpu)
            object_saliency_model.cuda(args.gpu)
            detr_model.cuda(args.gpu)
            audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
            object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model,
                                                                              device_ids=[args.gpu])

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best_{}.pth'.format(seed_num))
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best_{}.pth".format(seed_num))}')
    else:
        raise ValueError(print(
            f"Checkpoint not found: {ckp_fn}. Make sure the path is correct and 'args.experiment_name' and 'args.seed' are as same as your training phase."))


    # Transfrom to apply on the frame extracted directly from the video so that it can be processed like the dataloader data
    composed_transforms = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Set up directories
    temp_dir = os.path.join(args.data_dir, "temp")
    output_dir = os.path.join(args.data_dir, "processed_video")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    while True:
        # Record video
        video_file = record_video_with_audio(args.recording_duration, args.data_dir)

        # Load video and parameters
        video = mp.VideoFileClip(video_file)
        fps = video.fps
        duration = video.duration

        # Prepare to write the output processed video
        output_filename = os.path.join(output_dir, f"{video_file[-19:-4]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, heigth = 448, 448
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, heigth))
        processed_canvas = PredictionCanvas(width, heigth)

        # Divide video in segments for processing
        start_times = np.arange(0, duration, args.segment_duration)
        for start_time in start_times:

            # 1. NECESSARY INDICES AND FLOW CONTROL
            # Get other time points of the segment
            midpoint_time = start_time + (args.segment_duration / 2)
            end_time = min(start_time + args.segment_duration, duration)

            if end_time < start_time:
                break

            # 2. GET CENTRAL FRAMES AND AUDIO OF THE SEGMENT
            # File names
            central_frame_filename = os.path.join(temp_dir, f"{start_time}.jpg")
            audio_clip_filename = os.path.join(temp_dir, f"{start_time}.wav")

            # Extract central frame and save it as a temporary file
            central_frame = video.get_frame(midpoint_time)
            cv2.imwrite(central_frame_filename, cv2.cvtColor(central_frame, cv2.COLOR_RGB2BGR))

            # Extract audio for the segment and save it as a temporary file
            audio_clip = video.subclip(start_time, end_time).audio
            audio_clip.write_audiofile(audio_clip_filename, fps=44100, nbytes=2, codec='pcm_s16le')

            # 3. RUN MODEL
            testdataset = get_inference_dataset(start_time, temp_dir, temp_dir)
            testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            pred_avc, pred_og, pred_detr, pred_og_avc_detr = validate(testdataloader, audio_visual_model,
                                                                      object_saliency_model, detr_model, viz_dir, args)

            # 4. PROCESS OUTPUT FRAMES FOR THIS SEGMENT
            # Iterate over frames of current segment
            sub_clip = video.subclip(start_time, end_time)
            for frame in sub_clip.iter_frames():
                frame = Image.fromarray(np.uint8(frame))
                image = composed_transforms(frame)

                # Generate predictions
                img_pred_avc, img_pred_og, img_pred_detr, img_pred_og_avc_detr = get_visualization("name", image, pred_avc, pred_og, pred_detr, pred_og_avc_detr)

                # Generate prediction visualization
                processed_canvas.update_predictions(img_pred_avc, img_pred_og, img_pred_detr, img_pred_og_avc_detr)
                processed_canvas.create_canvas()
                out.write(processed_canvas.get_canvas())

        # Release the output video writer
        out.release()


        print("Press enter to record and process a new video.\nPress Q to terminate.")
        user_input = input()
        if user_input == 'q':
            delete_temp_folder(temp_dir)
            break
        else:
            continue

    return



def get_visualization(name, image, pred_av, pred_obj, pred_detr_800, pred_av_obj, viz_dir=None):
    denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    denorm_image = (denorm_image * 255).astype(np.uint8)
    if viz_dir != None:
        cv2.imwrite(os.path.join(viz_dir, f'{name}_image.jpg'), denorm_image)

    # visualize heatmaps
    heatmap_img = np.uint8(pred_av * 255)
    heatmap_pred_av = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin_pred_av = cv2.addWeighted(heatmap_pred_av, 0.8, np.uint8(denorm_image), 0.2, 0)
    if viz_dir != None:
        cv2.imwrite(os.path.join(viz_dir, f'{name}_AVC.jpg'), fin_pred_av)

    heatmap_img = np.uint8(pred_obj * 255)
    heatmap_pred_obj = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin_pred_obj = cv2.addWeighted(heatmap_pred_obj, 0.8, np.uint8(denorm_image), 0.2, 0)
    if viz_dir != None:
        cv2.imwrite(os.path.join(viz_dir, f'{name}_OG.jpg'), fin_pred_obj)

    heatmap_img_detr_800 = np.uint8(pred_detr_800 * 255)
    heatmap_detr_800 = cv2.applyColorMap(heatmap_img_detr_800[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin_detr_800 = cv2.addWeighted(heatmap_detr_800, 0.8, np.uint8(heatmap_detr_800), 0.2, 0)
    if viz_dir != None:
        cv2.imwrite(os.path.join(viz_dir, f'{name}_DETR.jpg'), fin_detr_800)

    heatmap_img = np.uint8(pred_av_obj * 255)
    heatmap_pred_av_obj = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
    fin_pred_av_obj  = cv2.addWeighted(heatmap_pred_av_obj, 0.8, np.uint8(denorm_image), 0.2, 0)
    if viz_dir != None:
        cv2.imwrite(os.path.join(viz_dir, f'{name}_OG_AVC_DETR.jpg'), fin_pred_av_obj )

    return fin_pred_av, fin_pred_obj, fin_detr_800, fin_pred_av_obj

@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, detr_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)


    for step, (image, detr_image, spec, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            detr_image = detr_image.cuda(args.gpu, non_blocking=True)

        measure_complexity = False
        if measure_complexity:
            from thop import profile
            #audio_visual model
            flops, params = profile(audio_visual_model, inputs=(image.float(),spec.float()))
            print(f"FLOPs of audio_visual_model: {flops/1000000000}")
            print(f"params of audio_visual_model: {params/1000000}")
            #Resnet18 OG model
            flops, params = profile(object_saliency_model, inputs=(image,))
            print(f"FLOPs of Resnet18 OG model: {flops/1000000000}")
            print(f"params Resnet18 OG model: {params/1000000}")
            #detr OG model
            flops, params = profile(detr_model, inputs=(detr_image,))
            print(f"FLOPs of detr model: {flops/1000000000}")
            print(f"params detr model: {params/1000000}")


        # Compute S_AVL
        img_f, aud_f = audio_visual_model(image.float(), spec.float())
        with torch.no_grad():
            Slogits = torch.einsum('nchw,mc->nmhw', img_f, aud_f) / args.tau
            Savl = Slogits[torch.arange(img_f.shape[0]), torch.arange(img_f.shape[0])]
            heatmap_av = Savl.unsqueeze(1)

        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()


        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        # Compute detr_OBJ
        detr_img_feat = get_detr_features(detr_model, detr_image)
        detr_heatmap_800 = F.interpolate(detr_img_feat, size=(800, 800), mode='bilinear', align_corners=True)
        detr_heatmap_800 = detr_heatmap_800.data.cpu().numpy()
        detr_heatmap_224 = F.interpolate(detr_img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        detr_heatmap_224 = detr_heatmap_224.data.cpu().numpy()

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_detr_224 = utils.normalize_img(detr_heatmap_224[i, 0])
            pred_detr_800 = utils.normalize_img(detr_heatmap_800[i, 0])
            # pred av obj for vggss
            pred_av_obj = utils.normalize_img(pred_av + args.alpha * pred_obj)
            if args.testset == "flickr":
                pred_av_obj = utils.normalize_img(pred_av/3 + pred_obj/3 + pred_detr_224/3)

            try:
                if args.save_visualizations:
                    get_visualization(name[i], image, pred_av, pred_obj, pred_detr_800, pred_av_obj, viz_dir)

            except KeyError as e:
                #print("ground truth bboxes for sample {} is not found.". format(name), e, bboxes)
                pass
    return pred_av, pred_obj, pred_detr_800, pred_av_obj

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


def get_detr_features(model, input_img):
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),
             model.transformer.encoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1])),
             model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_attn_weights.append(output[1]))]
    outputs = model(input_img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.3

    for hook in hooks:
        hook.remove()
    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    dert_dec_feats = dec_attn_weights.view(dec_attn_weights.size()[1], h, w)
    if len(keep.nonzero()) == 1:
        feats = dert_dec_feats[keep.nonzero()[0][0]].unsqueeze(0).unsqueeze(0)
    else:  # len(keep.nonzero()[0]) > 1:
        strong_featues = torch.zeros((len(keep.nonzero())), dert_dec_feats.size()[1], dert_dec_feats.size()[2])
        for idx, index in enumerate(keep.nonzero()):
            one_feat = dert_dec_feats[index[0]]
            strong_featues[idx, :, :] = one_feat
        feats = strong_featues.abs().mean(dim=0).unsqueeze(0).unsqueeze(0)
    return feats

if __name__ == "__main__":
    main(get_arguments())

