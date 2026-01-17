# ComfyUI_Flux2ImageReference

A comprehensive node for ComfyUI that implements an image reference mechanism for Flux2-based image generation models. This node allows you to incorporate visual features from reference images into your text-conditioned image generation process.

## Features 特性

- Accept up to 5 reference images as input 最多接受5张参考图像作为输入
- Encode images using selected VAE into reference latents 使用选定的VAE将图像编码为参考潜空间
- Combine visual features from reference latents with text conditioning 将参考潜空间的视觉特征与文本条件相结合
- Dynamic fusion of visual and textual features for enhanced generation 动态融合视觉和文本特征以增强生成效果
- Adjustable strength parameter to control influence of reference images 可调节强度参数以控制参考图像的影响
<img src="https://github.com/naku-yh/ComfyUI_Flux2ImageReference/blob/main/pic_1/Simple%20Workflow.png" alt="FLUX2 Image Reference Node Example" width="500"/>


## Installation 安装

1. Clone this repository into your ComfyUI `custom_nodes` directory:
将此仓库克隆到您的 ComfyUI `custom_nodes` 目录中：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/naku-yh/ComfyUI_Flux2ImageReference.git
```

2. Restart ComfyUI
重启 ComfyUI

## Usage 使用方法
<img src="https://github.com/naku-yh/ComfyUI_Flux2ImageReference/blob/main/pic_1/SimpleNode.png" alt="FLUX2 Image Reference Node Example" width="500"/>
### Node Inputs 节点输入

- `vae`: VAE model to encode reference images (connect from a VAE loader node)
       VAE模型用于编码参考图像（从VAE加载器节点连接）
- `strength`: Control the influence of reference image features (0.0-2.0)
              控制参考图像特征的影响（0.0-2.0）
- `image1-image5`: Up to 5 reference images to extract visual features from
                  最多5张用于提取视觉特征的参考图像
- `conditioning`: Optional text conditioning to combine with visual features
                 可选的文本条件与视觉特征相结合

### Node Output 节点输出

- `CONDITIONING`: Enhanced conditioning that incorporates both text and visual features
                增强的条件，包含文本和视觉特征

### Workflow 工作流程

1. Load your VAE model using a standard ComfyUI VAE loader node
   使用标准的ComfyUI VAE加载器节点加载VAE模型
2. Load your reference images using standard ComfyUI image loader nodes
   使用标准的ComfyUI图像加载器节点加载参考图像
3. Connect the images to the corresponding inputs (image1-image5)
   将图像连接到相应的输入（image1-image5）
4. Optionally connect text conditioning from CLIP text encoder
   可选择性地连接来自CLIP文本编码器的文本条件
5. Connect the VAE to the vae input
   将VAE连接到vae输入
6. Adjust the strength parameter to control how much the reference images influence the generation
   调整强度参数以控制参考图像对生成的影响程度
7. Connect the output conditioning to your sampling node
   将输出条件连接到您的采样节点

## Technical Details 技术细节

The node works by:
节点工作原理：

1. Encoding each reference image through the selected VAE to obtain reference latents
   通过选定的VAE对每个参考图像进行编码以获得参考潜空间
2. Extracting statistical features (mean, std, min, max) from the reference latents
   从参考潜空间中提取统计特征（均值、标准差、最小值、最大值）
3. Injecting these visual features into the text conditioning through feature fusion
   通过特征融合将这些视觉特征注入文本条件
4. Producing a combined conditioning that contains both textual and visual information
   生成包含文本和视觉信息的组合条件

## Notes 注意事项

- VAE is provided as a separate input connection, allowing for more flexibility
  VAE作为单独的输入连接提供，提供更多灵活性
- Higher strength values will result in more influence from reference images
  较高的强度值将导致参考图像产生更多影响
- When no conditioning is provided, the node creates conditioning based solely on reference latents
  当未提供条件时，节点仅基于参考潜空间创建条件
- Compatible with Flux2-based models and other diffusion models that accept conditioning
  兼容基于Flux2的模型和接受条件的其他扩散模型

## License 许可证

MIT
