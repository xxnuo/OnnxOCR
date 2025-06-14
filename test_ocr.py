import argparse
import os
import time

import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from onnxocr.utils import draw_ocr


def parse_args():
    parser = argparse.ArgumentParser(description="OCR大图像处理工具")
    parser.add_argument(
        "--image", type=str, default="./test_images/web2.png", help="输入图像路径"
    )
    parser.add_argument("--max_size", type=int, default=2048, help="每个分块的最大尺寸")
    parser.add_argument("--overlap", type=int, default=200, help="分块之间的重叠像素")
    parser.add_argument("--memory_limit", type=int, default=2048, help="内存限制(MB)")
    parser.add_argument("--use_gpu", action="store_true", help="是否使用GPU")
    parser.add_argument(
        "--vis_width", type=int, default=1920, help="可视化图像的最大宽度"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出图像路径，默认为result_img/large_image_result_时间戳.jpg",
    )
    return parser.parse_args()


def sav2Img(
    org_img, result, name="draw_ocr.jpg", resize_for_vis=False, max_vis_width=1920
):
    """
    将OCR结果可视化并保存到图像

    Args:
        org_img: 原始图像
        result: OCR结果
        name: 输出图像文件名
        resize_for_vis: 是否调整图像大小以便于可视化
        max_vis_width: 可视化图像的最大宽度
    """
    # 显示结果
    import os

    from PIL import Image

    # 确保输出目录存在
    output_dir = os.path.dirname(name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    # 图像转BGR2RGB
    image = org_img[:, :, ::-1]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    # 对于大图像，可以选择调整大小以便于可视化
    im_show = draw_ocr(
        image,
        boxes,
        txts,
        scores,
        resize_img_for_vis=resize_for_vis,
        input_size=max_vis_width,
    )
    im_show = Image.fromarray(im_show)
    im_show.save(name)


def main():
    # 解析命令行参数
    args = parse_args()

    # 创建结果目录
    os.makedirs("result_img", exist_ok=True)

    # 初始化模型
    model = ONNXPaddleOcr(use_angle_cls=True, use_gpu=args.use_gpu)

    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"错误：无法读取图像 {args.image}")
        return

    print(f"图像大小: {img.shape[1]}x{img.shape[0]}")

    # 开始处理
    s = time.time()
    # 使用新的大图像处理方法，自动适应图像大小
    result = model.ocr_large_image(img)
    e = time.time()
    print("总处理时间: {:.3f}秒".format(e - s))
    print(f"识别出的文本数量: {len(result[0])}")

    # 确定输出文件名
    if args.output:
        output_filename = args.output
    else:
        output_filename = f"result_img/large_image_result_{int(time.time())}.jpg"

    # 保存结果
    sav2Img(
        img,
        result,
        name=output_filename,
        resize_for_vis=True,
        max_vis_width=args.vis_width,
    )
    print(f"结果已保存到: {output_filename}")

    # 输出前10个识别结果示例
    print("\n识别结果:")
    for i, box in enumerate(result[0][: min(10, len(result[0]))]):
        print(f"{i + 1}. {box[1][0]} (置信度: {box[1][1]:.2f})")


if __name__ == "__main__":
    main()
