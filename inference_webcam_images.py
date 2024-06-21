"""
Audio-Visual Inference Script

This script modifies the script test_N_times.py to support inference
using the laptop's webcam and microphone.

Modifications from original:
- Added functionality to capture audio from microphone and images from webcam.
- Implemented loop to continuously capture and analyze audio-visual data.
- Removed functions that relied on ground truths and evaluation metrics

Usage:
- $ python inference_webcam_images.py --experiment_name test_run1 [--model_dir checkpoints] [--recording_duration 3]

Notes:
- Output is saved at og_avc/checkpoints/run_name/viz_seed_10
"""

import os
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
    parser.add_argument('--recording_duration', default=3.0, type=float, help="Duration of audio recording in seconds")

    return parser.parse_args()


def main(args):
    """
    Inference loop with audio-visual data.

    Parameters:
    - args (argparse.Namespace)
    """

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

    # Set up recording directories
    audio_dir = os.path.join('data','audio')
    frames_dir = os.path.join('data', 'frames')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Record audio and image and process them until user interrupt execution
    while True:
        # Generate unique filenames for audio and image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        audio_filename = os.path.join(audio_dir, f"{timestamp}.wav")
        image_filename = os.path.join(frames_dir, f"{timestamp}.jpg")

        # Create threads for recording audio and capturing image
        audio_thread = threading.Thread(target=record_audio, args=(audio_filename, args.recording_duration))
        image_thread = threading.Thread(target=capture_image, args=(image_filename,))

        # Start both threads
        audio_thread.start()
        image_thread.start()

        # Wait for both threads to finish
        audio_thread.join()
        image_thread.join()

        print(f"Audio and image saved: {audio_filename}, {image_filename}")

        testdataset = get_inference_dataset(timestamp, audio_dir, frames_dir)
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        print("Loaded dataloader.")

        validate_inference(testdataloader, audio_visual_model, object_saliency_model, detr_model, viz_dir, args)

        print("Press enter to record and process a new input.\nPress Q to terminate.")
        user_input = input()
        if user_input == 'q':
                break
        else:
            continue
    return


@torch.no_grad()
def validate_inference(testdataloader, audio_visual_model, object_saliency_model, detr_model, viz_dir, args):
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
            #pred av obj for vggss
            pred_av_obj = utils.normalize_img(pred_av + args.alpha * pred_obj)
            if args.testset == "flickr":
                pred_av_obj = utils.normalize_img(pred_av/3 + pred_obj/3 + pred_detr_224/3)

            try:
                if args.save_visualizations:
                    denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                    denorm_image = (denorm_image*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_image.jpg'), denorm_image)

                    # visualize heatmaps
                    heatmap_img = np.uint8(pred_av*255)
                    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_AVC.jpg'), fin)

                    heatmap_img = np.uint8(pred_obj*255)
                    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_OG.jpg'), fin)

                    heatmap_img_detr_800 = np.uint8(pred_detr_800 * 255)
                    heatmap_img_detr_800 = cv2.applyColorMap(heatmap_img_detr_800[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin_detr_800 = cv2.addWeighted(heatmap_img_detr_800, 0.8, np.uint8(heatmap_img_detr_800), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_DETR.jpg'), fin_detr_800)

                    heatmap_img = np.uint8(pred_av_obj*255)
                    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_OG_AVC_DETR.jpg'), fin)
            except KeyError as e:
                #print("ground truth bboxes for sample {} is not found.". format(name), e, bboxes)
                pass
    return

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

