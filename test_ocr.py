import argparse
import os
import sys
import time

import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img


def parse_args():
    parser = argparse.ArgumentParser(description="OCR大图像处理工具")
    parser.add_argument("--image", type=str, default="./web.png", help="输入图像路径")
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
    result = model.ocr_large_image(
        img,
        max_size=args.max_size,
        overlap=args.overlap,
        memory_limit_mb=args.memory_limit,
    )
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
    print("\n前10个识别结果示例:")
    for i, box in enumerate(result[0][: min(10, len(result[0]))]):
        print(f"{i + 1}. {box[1][0]} (置信度: {box[1][1]:.2f})")


if __name__ == "__main__":
    main()
