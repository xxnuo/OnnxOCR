import argparse
import gc
import os
import sys
import time

import numpy as np

from .predict_system import TextSystem
from .utils import draw_ocr, str2bool
from .utils import infer_args as init_args


class ONNXPaddleOcr(TextSystem):
    def __init__(self, **kwargs):
        # 默认参数
        parser = init_args()
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        params = argparse.Namespace(**inference_args_dict)

        # params.rec_image_shape = "3, 32, 320"
        params.rec_image_shape = "3, 48, 320"

        # 根据传入的参数覆盖更新默认参数
        params.__dict__.update(**kwargs)

        # 初始化模型
        super().__init__(params)

    def ocr_large_image(
        self,
        img,
        max_size=2560,
        overlap=200,
        det=True,
        rec=True,
        cls=True,
        memory_limit_mb=1024,
    ):
        """
        处理超大图像的方法，通过分块处理来避免信息丢失和内存溢出

        Args:
            img: 输入图像
            max_size: 每个分块的最大尺寸
            overlap: 分块之间的重叠像素
            det: 是否进行文本检测
            rec: 是否进行文本识别
            cls: 是否使用方向分类器
            memory_limit_mb: 每个分块处理的内存限制（MB）

        Returns:
            合并后的OCR结果
        """
        h, w = img.shape[:2]

        # 计算图像大小（以MB为单位）
        img_size_mb = img.nbytes / (1024 * 1024)

        # 根据内存限制动态调整分块大小
        if img_size_mb > memory_limit_mb:
            # 计算缩放比例
            scale_factor = np.sqrt(memory_limit_mb / img_size_mb)
            # 调整max_size
            adjusted_max_size = int(max_size * scale_factor)
            # 确保max_size不会太小
            max_size = max(adjusted_max_size, 1024)

        # 如果图像尺寸不大，直接使用原始方法处理
        if max(h, w) <= max_size:
            return self.ocr(img, det, rec, cls)

        # 分块处理
        result_boxes = []
        result_texts = []

        # 计算需要分成多少块
        h_blocks = max(1, (h - overlap) // (max_size - overlap) + 1)
        w_blocks = max(1, (w - overlap) // (max_size - overlap) + 1)

        # 如果只需要一个块，直接处理
        if h_blocks == 1 and w_blocks == 1:
            return self.ocr(img, det, rec, cls)

        # 计算每个块的实际大小
        h_step = (h - overlap) // h_blocks + overlap
        w_step = (w - overlap) // w_blocks + overlap

        print(
            f"处理大图像: {h}x{w}，分为 {h_blocks}x{w_blocks} 个块，每块大小约为 {h_step}x{w_step}"
        )

        total_blocks = h_blocks * w_blocks
        processed_blocks = 0

        for i in range(h_blocks):
            for j in range(w_blocks):
                # 计算当前块的坐标
                y1 = max(0, i * (h_step - overlap))
                y2 = min(h, y1 + h_step)
                x1 = max(0, j * (w_step - overlap))
                x2 = min(w, x1 + w_step)

                # 提取当前块
                block = img[y1:y2, x1:x2].copy()  # 使用copy()避免引用原始大图

                processed_blocks += 1
                print(
                    f"处理块 {processed_blocks}/{total_blocks}: 坐标 ({x1}, {y1}) 到 ({x2}, {y2}), 大小 {block.shape[1]}x{block.shape[0]}"
                )

                # 处理当前块
                block_result = self.ocr(block, det, rec, cls)

                if not block_result or not block_result[0]:
                    continue

                # 调整坐标到原图
                for box_info in block_result[0]:
                    box = box_info[0]
                    text_info = box_info[1]

                    # 调整坐标
                    adjusted_box = []
                    for point in box:
                        adjusted_box.append([point[0] + x1, point[1] + y1])

                    result_boxes.append(adjusted_box)
                    result_texts.append(text_info)

                # 释放内存
                del block
                gc.collect()

        # 合并结果
        merged_result = []
        merged_result.append(
            [[box, text] for box, text in zip(result_boxes, result_texts)]
        )

        # 去除重复检测
        if len(merged_result[0]) > 0:
            merged_result[0] = self._remove_duplicates(merged_result[0])

        return merged_result

    def _remove_duplicates(self, results, iou_threshold=0.5):
        """
        移除重复的检测结果

        Args:
            results: OCR结果列表
            iou_threshold: IOU阈值，超过此阈值的框被认为是重复的

        Returns:
            去重后的结果列表
        """
        if len(results) <= 1:
            return results

        # 提取所有框和对应的文本
        boxes = [r[0] for r in results]
        texts = [r[1] for r in results]

        # 计算每个框的面积
        areas = []
        for box in boxes:
            box_np = np.array(box)
            x_min = np.min(box_np[:, 0])
            y_min = np.min(box_np[:, 1])
            x_max = np.max(box_np[:, 0])
            y_max = np.max(box_np[:, 1])
            areas.append((x_max - x_min) * (y_max - y_min))

        # 按面积从大到小排序
        indices = np.argsort(areas)[::-1]

        # 去重
        keep = []
        for i in range(len(indices)):
            idx1 = indices[i]
            if idx1 < 0:
                continue

            keep.append(idx1)
            box1 = np.array(boxes[idx1])

            # 计算box1的边界
            x1_min = np.min(box1[:, 0])
            y1_min = np.min(box1[:, 1])
            x1_max = np.max(box1[:, 0])
            y1_max = np.max(box1[:, 1])

            for j in range(i + 1, len(indices)):
                idx2 = indices[j]
                if idx2 < 0:
                    continue

                box2 = np.array(boxes[idx2])

                # 计算box2的边界
                x2_min = np.min(box2[:, 0])
                y2_min = np.min(box2[:, 1])
                x2_max = np.max(box2[:, 0])
                y2_max = np.max(box2[:, 1])

                # 计算交集面积
                x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                intersection = x_overlap * y_overlap

                # 计算并集面积
                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)
                union = area1 + area2 - intersection

                # 计算IOU
                iou = intersection / union if union > 0 else 0

                # 如果IOU大于阈值，并且文本相似度高，则认为是重复的
                if (
                    iou > iou_threshold
                    and self._text_similarity(texts[idx1][0], texts[idx2][0]) > 0.7
                ):
                    indices[j] = -1  # 标记为已处理

        # 返回保留的结果
        return [results[i] for i in keep]

    def _text_similarity(self, text1, text2):
        """
        计算两个文本的相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度得分，范围[0, 1]
        """
        # 如果两个文本完全相同
        if text1 == text2:
            return 1.0

        # 如果一个文本是另一个的子串
        if text1 in text2 or text2 in text1:
            shorter = min(len(text1), len(text2))
            longer = max(len(text1), len(text2))
            return shorter / longer

        # 计算编辑距离
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # 简单的相似度计算：共同字符数 / 总字符数
        common_chars = set(text1) & set(text2)
        return len(common_chars) / len(set(text1) | set(text2))

    def ocr(self, img, det=True, rec=True, cls=True):
        if cls and not self.use_angle_cls:
            print(
                "Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process"
            )

        if det and rec:
            ocr_res = []
            dt_boxes, rec_res = self.__call__(img, cls)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            dt_boxes = self.text_detector(img)
            tmp_res = [box.tolist() for box in dt_boxes]
            ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []

            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res_tmp = self.text_classifier(img)
                if not rec:
                    cls_res.append(cls_res_tmp)
            rec_res = self.text_recognizer(img)
            ocr_res.append(rec_res)

            if not rec:
                return cls_res
            return ocr_res
