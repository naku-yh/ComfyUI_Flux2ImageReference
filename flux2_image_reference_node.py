import torch
import os
import comfy.model_management as model_management


class NakuNode_Flux2:
    """
    NakuNode Flux2 Image Reference Node for Flux2-based Image Generation
    Takes up to 5 images, encodes them with selected VAE into reference latents,
    and combines with text conditioning for feature fusion.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "conditioning": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_reference"
    CATEGORY = "NakuNode_Flux2"

    def apply_reference(self, vae, strength, image1=None, image2=None, image3=None, image4=None, image5=None, conditioning=None):
        # 收集图像
        images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

        if not images:
            raise ValueError("At least one image is required")

        # 对每张图像进行VAE编码
        reference_latents = []
        for i, image in enumerate(images):
            # 确保图像维度正确
            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            # 验证图像通道数，确保是RGB或RGBA
            if image.shape[-1] < 3:
                raise ValueError(f"Image {i+1} has insufficient channels: {image.shape[-1]}")

            # 编码图像为latent
            # 将图像移动到适当的设备上
            device = model_management.get_torch_device()
            image = image.to(device)

            # 编码图像为latent
            # 注意：VAE的encode方法期望输入形状为(batch, height, width, channel)，值范围为[0,1]
            # 并且通常只取前3个通道(RGB)
            image_rgb = image[:, :, :, :3]
            latent = vae.encode(image_rgb)
            reference_latents.append(latent)

        # 如果有原始conditioning，则与参考latent结合
        if conditioning is not None:
            # 这里需要实现将reference_latents特征注入到conditioning中的逻辑
            # 由于ComfyUI的conditioning格式复杂，这里简化处理
            # 实际实现可能需要更复杂的attention操作
            processed_conditioning = self.inject_features(conditioning, reference_latents, strength)
        else:
            # 如果没有原始conditioning，返回基于参考latent的新conditioning
            processed_conditioning = self.create_conditioning_from_latents(reference_latents, strength)

        return (processed_conditioning,)

    def inject_features(self, conditioning, reference_latents, strength):
        """
        将参考latent的特征注入到原始conditioning中
        这是一个简化的实现，实际需要根据具体模型架构调整
        """
        # 在Flux2模型中，conditioning通常包含文本嵌入和其他相关信息
        # 我们需要将参考latent的信息融入到conditioning中
        # 这里我们采用一种简单的方式，将latent的统计信息附加到conditioning中

        # 获取device信息
        device = model_management.get_torch_device()

        # 计算所有参考latents的平均统计信息
        combined_latent_stats = []
        for latent in reference_latents:
            # 计算latent的统计信息，如均值、方差等
            # 这些统计信息可以作为视觉特征的表示
            latent_mean = latent.mean(dim=[1, 2, 3], keepdim=True)  # (batch, channels, 1, 1)
            latent_std = latent.std(dim=[1, 2, 3], keepdim=True)    # (batch, channels, 1, 1)

            # 将统计信息展平以便后续处理
            stats = torch.cat([latent_mean.squeeze(-1).squeeze(-1),
                              latent_std.squeeze(-1).squeeze(-1)], dim=-1)  # (batch, 2*channels)
            combined_latent_stats.append(stats)

        # 计算所有latents的平均统计信息
        if len(combined_latent_stats) > 1:
            avg_latent_stats = torch.stack(combined_latent_stats, dim=0).mean(dim=0)
        else:
            avg_latent_stats = combined_latent_stats[0]

        # 创建新的conditioning，将latent特征注入到原始conditioning中
        new_conditioning = []
        for cond in conditioning:
            # 复制原始条件，安全地处理可能包含不同数据类型的列表
            n = []
            for item in cond:
                if torch.is_tensor(item):
                    n.append(torch.clone(item))
                else:
                    # 对于非张量类型，直接复制引用
                    n.append(item)

            # 尝试将latent统计信息注入到conditioning中
            # conditioning[0]通常是文本嵌入张量，尝试将其与latent统计信息结合
            if len(n) > 0 and torch.is_tensor(n[0]) and n[0].dim() >= 2:
                # 获取原始conditioning的形状信息
                orig_cond = n[0]

                # 调整avg_latent_stats的大小以匹配orig_cond的形状
                # 这里假设orig_cond的形状是(batch, seq_len, features)
                batch_size = orig_cond.shape[0]

                # 重复latent统计信息以匹配批次大小
                expanded_latent_stats = avg_latent_stats.repeat(batch_size, 1)

                # 为了匹配orig_cond的序列长度维度，我们需要扩展
                # 一种方法是将latent_stats附加到每个序列元素
                seq_len = orig_cond.shape[1] if orig_cond.dim() > 1 else 1
                features_per_token = orig_cond.shape[-1] if orig_cond.dim() > 1 else orig_cond.shape[-1]

                # 简单的方法：将latent统计信息添加到原始conditioning中
                # 这里我们将其作为一个额外的token或直接相加（取决于模型需求）
                # 为简单起见，我们将latent特征添加到现有特征上
                if orig_cond.shape[0] == expanded_latent_stats.shape[0]:
                    # 调整expanded_latent_stats以匹配orig_cond的最后一个维度
                    if expanded_latent_stats.shape[1] != orig_cond.shape[-1]:
                        # 如果维度不匹配，尝试截断或填充
                        target_dim = orig_cond.shape[-1]
                        if expanded_latent_stats.shape[1] > target_dim:
                            expanded_latent_stats = expanded_latent_stats[:, :target_dim]
                        else:
                            # 填充到目标维度
                            padding_size = target_dim - expanded_latent_stats.shape[1]
                            expanded_latent_stats = torch.cat([
                                expanded_latent_stats,
                                torch.zeros(expanded_latent_stats.shape[0], padding_size).to(expanded_latent_stats.device)
                            ], dim=1)

                    # 应用强度参数
                    injected_cond = orig_cond + strength * expanded_latent_stats.unsqueeze(1)
                    n[0] = injected_cond

            # 在conditioning的字典部分添加reference_latents信息
            # 检查cond中是否有字典类型的元素（通常在cond[1]位置）
            for i, item in enumerate(cond):
                if isinstance(item, dict):
                    # 找到了字典，向其中添加reference_latents
                    n[i]['reference_latents'] = reference_latents
                    break
            else:
                # 如果没有找到字典，创建一个新的字典并添加到列表中
                n.append({'reference_latents': reference_latents})

            new_conditioning.append(n)

        return new_conditioning

    def create_conditioning_from_latents(self, reference_latents, strength):
        """
        基于参考latent创建新的conditioning
        """
        # 获取device信息
        device = model_management.get_torch_device()

        # 计算所有参考latents的综合特征表示
        combined_latent_features = []
        for latent in reference_latents:
            # 计算latent的多级统计信息作为特征表示
            latent_mean = latent.mean(dim=[2, 3], keepdim=True)  # (batch, channels, 1, 1)
            latent_std = latent.std(dim=[2, 3], keepdim=True)    # (batch, channels, 1, 1)
            latent_max = latent.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]  # (batch, channels, 1, 1)
            latent_min = latent.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]  # (batch, channels, 1, 1)

            # 组合多种统计信息
            combined_stats = torch.cat([
                latent_mean,
                latent_std,
                latent_max,
                latent_min
            ], dim=1)  # (batch, 4*channels, 1, 1)

            # 展平并应用强度
            flattened_stats = combined_stats.view(latent.shape[0], -1) * strength  # (batch, 4*channels)
            combined_latent_features.append(flattened_stats)

        # 合并所有图像的特征
        if len(combined_latent_features) > 1:
            # 可以选择平均、拼接或其他融合策略
            # 这里使用平均作为融合策略
            fused_features = torch.stack(combined_latent_features, dim=0).mean(dim=0)
        else:
            fused_features = combined_latent_features[0]

        # 创建模拟的conditioning结构
        # 在实际应用中，这需要根据具体模型的要求进行调整
        # 这里我们创建一个与Flux2兼容的conditioning格式
        c = []

        # 创建一个模拟的文本嵌入，其中包含从图像中提取的视觉特征
        # 假设conditioning的基本格式为[嵌入向量, 其他参数字典]
        batch_size = fused_features.shape[0]

        # 创建一个模拟的嵌入向量，其维度与模型期望的文本嵌入一致
        # 由于我们不知道确切的维度，我们创建一个合理的近似
        embedding_dim = 768  # 假设使用类似CLIP的768维嵌入
        if fused_features.shape[1] > embedding_dim:
            # 如果特征过多，截断
            simulated_embedding = fused_features[:, :embedding_dim].unsqueeze(1)  # (batch, 1, embedding_dim)
        elif fused_features.shape[1] < embedding_dim:
            # 如果特征不足，填充零
            padding_needed = embedding_dim - fused_features.shape[1]
            padding = torch.zeros(batch_size, padding_needed).to(device)
            padded_features = torch.cat([fused_features, padding], dim=1)
            simulated_embedding = padded_features.unsqueeze(1)  # (batch, 1, embedding_dim)
        else:
            simulated_embedding = fused_features.unsqueeze(1)  # (batch, 1, embedding_dim)

        # 创建conditioning条目，包含reference_latents信息
        conditioning_entry = [simulated_embedding, {"pooled_output": fused_features, "reference_latents": reference_latents}]  # 添加额外的池化输出和参考latents
        c.append(conditioning_entry)

        return c