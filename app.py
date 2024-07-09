import gradio as gr
import argparse 
import torch
import cv2
import numpy as np
import torch
import argparse
from src.helpers import init_helper, vsumm_helper, bbox_helper, video_helper
from src.modules.model_get import get_model


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument('--model', type=str,
                        choices=('anchor-based', 'anchor-free'), default='anchor-based')
    parser.add_argument('--dataset', type=str, default='tvsum')
    # training & evaluation
    parser.add_argument('--device', type=str, default='cpu',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=[])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='../models/model')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--nms-thresh', type=float, default=0.5)

    # inference
    parser.add_argument('--ckpt-path', type=str, default="checkpoint/tvsum.yml.0.pt")
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--source', type=str, default="input_video.mp4")
    parser.add_argument('--save-path', type=str, default="output.mp4")

    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)

    # anchor based
    parser.add_argument('--neg-sample-ratio', type=float, default=2.0)
    parser.add_argument('--incomplete-sample-ratio', type=float, default=1.0)
    parser.add_argument('--pos-iou-thresh', type=float, default=0.6)
    parser.add_argument('--neg-iou-thresh', type=float, default=0.0)
    parser.add_argument('--incomplete-iou-thresh', type=float, default=0.3)
    parser.add_argument('--anchor-scales', type=int, nargs='+',
                        default=[4, 8, 16, 32])

    # anchor free
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    parser.add_argument('--cls-loss', type=str, default='focal',
                        choices=['focal', 'cross-entropy'])
    parser.add_argument('--reg-loss', type=str, default='soft-iou',
                        choices=['soft-iou', 'smooth-l1'])

    return parser


def main(model):
    parser  =  get_parser()
    args = parser.parse_args()
    if model == 'Anchor-based': 
        args.model = 'anchor-based'
        args.ckpt_path = 'checkpoints/tvsum_ab.ckpt'
        
    else: 
        args.model = 'anchor-free'
        args.ckpt_path = 'checkpoints/tvsum_af.ckpt'
    
    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    state_dict = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # load video
    print('Preprocessing source video ...')
    video_proc = video_helper.VideoPreprocessor(args.sample_rate)
    n_frames, seq, cps, nfps, picks = video_proc.run(args.source)
    seq_len = len(seq)

    print('Predicting summary ...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)

        pred_cls, pred_bboxes = model.predict(seq_torch)

        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)
        pred_summ = vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

    print('Writing summary video ...')

    # load original video
    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create summary video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if pred_summ[frame_idx]:
            out.write(frame)

        frame_idx += 1

    out.release()
    cap.release()


import cv2

def write_mp4(input_file, output_file):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (H264 in this case)
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # Write the frame into the output file
            out.write(frame)
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()



def video_identity(video, model):
    write_mp4(video, 'input_video.mp4')
    main(model)
    
    return 'output.mp4'

model_options = ['Anchor-based', 'Anchor-free'] 

demo = gr.Interface(video_identity, 
                     [gr.Video(), gr.Dropdown(choices=model_options, label="Select Model")],
                    "playable_video",
                     title="Video Summarization"
                    )

if __name__ == "__main__":
    demo.launch()