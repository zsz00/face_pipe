# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import mmcv
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument('--out', type=str, help='Output video file')
    args = parser.parse_args()
    return args


def show_result_pyplot(model, img, result, score_thr=0.3, title='result', wait_time=0, out_file=''):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        font_size=9,
        out_file=out_file)


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr, out_file=args.out)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)


"""
conda activate open-mmlab
python demo/image_demo.py demo/test_data/5.jpg configs/yolo/yolov3_d53_mstrain-608_273e_face_1.py \
/mnt/zy_data/detection/face_detection/out_2/latest.pth --out=demo/test_data/5_out_2_6.jpg --score-thr=0.2

python demo/image_demo.py demo/test_data/5.jpg configs/yolo/yolov3_d53_mstrain-608_273e_face_1.py \
/mnt/zy_data/detection/face_detection/out_2/epoch_2.pth --out=demo/test_data/5_out_2_1.jpg --score-thr=0.2

python demo/image_demo.py demo/test_data/7_Cheering_Cheering_7_57.jpg configs/yolox/yolox_s_8x8_300e_face_1.py \
/mnt/zy_data/detection/face_detection/out_3/latest.pth --out=demo/test_data/7_out_3_1.jpg --score-thr=0.2

python demo/image_demo.py demo/test_data/0_Parade_marchingband_1_17.jpg configs/yolox/yolox_s_8x8_300e_face_1.py \
/mnt/zy_data/detection/face_detection/out_3/latest.pth --out=demo/test_data/0_out_3_1.jpg --score-thr=0.2

python demo/image_demo.py demo/test_data/0.jpg configs/yolox/yolox_s_8x8_300e_face_1.py \
/mnt/zy_data/detection/face_detection/out_3/latest.pth --out=demo/test_data/0_out_3_8.jpg --score-thr=0.3

python demo/image_demo.py demo/test_data/0.jpg configs/yolox/yolox_s_8x8_300e_face_1.py \
/mnt/zy_data/detection/face_detection/out_4/latest.pth --out=demo/test_data/0_out_4_3.jpg --score-thr=0.3

"""
