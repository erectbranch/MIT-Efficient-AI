<div width="100%" height="100%" align="center">
  
<h1 align="center">
  <p align="center">TinyML and Efficient Deep Learning Computing</p>
</h1>

<b>MIT 6.S965/6.5940 • Fall • 2022-2024</b>
<br>
Instructor : Song Han(Associate Professor, MIT EECS)

</div>

<br>

Lecture notes for courses [MIT 6.S965, Fall 2022 | MIT 6.5940, Fall 2023•2024](https://efficientml.ai)

## Courses

| Course | Video | Slide | Note | Homework |
| --- | --- | --- | --- | --- |
| MIT 6.5940 • 2024 • Fall | [Videos](https://youtube.com/playlist?list=PL80kAHvQbh-qGtNc54A6KW4i4bkTPjiRF) | [Slides](https://hanlab.mit.edu/courses/2024-fall-65940#schedule) | [Notes](2024/) | [Lab 1](https://colab.research.google.com/drive/1Fagq3JQBzCizodyxpHKvWDzfCC7F1RWN) / [Lab 2](https://colab.research.google.com/drive/11IBla1q1McoZ2oCANCGHns8VtzG5nCMP) / [Lab 3](https://colab.research.google.com/drive/1xKReLBHVS6bkFbYkfi-Ky3C4loQmG6Yc) / [Lab 4](https://colab.research.google.com/drive/16H9RvSg4XIF35X3fLGQUVwAE9ccvDj14) / [Lab 5](https://drive.google.com/drive/folders/1MhMvxvLsyYrN-4C6eQG8Zj2JeSuyAOf0) |
| MIT 6.5940 • 2023 • Fall | [Videos](https://youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB) | [Slides](https://hanlab.mit.edu/courses/2023-fall-65940#schedule) | [Notes](2023/) | - |
| MIT 6.S965 • 2022 • Fall | [Videos](https://youtube.com/playlist?list=PL80kAHvQbh-ocildRaxjjBy6MR1ZsNCU7) | [Slides](https://hanlab.mit.edu/courses/2022-fall-6s965#schedule) | [Notes](2022/) | [Lab 4: Deployment on MCU](https://github.com/Xuweijia-buaa/MIT-6.S965-TinyML-and-Efficient-Deep-Learning-Computing/blob/main/notebooks/mit_6s965_lab4_tinyml.ipynb) |

## Lecture Notes

### 📖 Basics of Deep Learning

- [Basic Terminologies, Shape of Tensors](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec02/summary01)

  > Synapse(weight), Neuron(activation), Cell body

  > Fully-Connected layer, Convolution layer(padding, stride, receptive field, grouped convolution), Pooling layer

- [Efficiency Metrics](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec02/summary02)

  > Metrics(latency, storage, energy)

  > Memory-Related(\#parameters, model size, \#activations), Computation(MACs, FLOP)

### 📙 Efficient Inference

- [Pruning Granularity, Pruning Critertion](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec03)

  > Unstructured/Structured pruning(Fine-grained/Pattern-based/Vector-level/Kernel-level/Channel-level)
  
  > Pruning Criterion: Magnitude(L1-norm, L2-norm), Sensitivity and Saliency(SNIP), Loss Change(First-Order, Second-Order Taylor Expansion)
  
  > Data-Aware Pruning Criterion: Average Percentage of Zero(APoZ), Reconstruction Error, Entropy

- [Automatic Pruning, Lottery Ticket Hypothesis](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec04/summary01)

  > Finding Pruning Ratio: Reinforcement Learning based, Rule based, Regularization based, Meta-Learning based 

  > Lottery Ticket Hypothesis(Winning Ticket, Iterative Magnitude Pruning, Scaling Limitation)

  > Pruning at Initialization(Connection Sensitivity, Gradient Flow)

- [System & Hardware Support for Fine-grained Sparsity](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec04/summary02)

  > Efficient Inference Engine(EIE format: relative index, column pointer)

- [Sparse Matrix-Matrix Multiplication, GPU Support for Sparsity](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec04/summary03)

  > Sparse Matrix-Matrix Multiplication(SpMM), CSR format

  > GPU Support for Sparsity: Hierarchical 1-Dimensional Tiling, Row Swizzle, M:N Sparsity, Block SpMM(Blocked-ELL format), PatDNN(FKW format)

  ---

- [Basic Concepts of Quantization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec05/summary01)

  > Numeric Data Types: Integer, Fixed-Point, Floating-Point(IEEE FP32/FP16, BF16, NVIDIA FP8), INT4 and FP4

  > Uniform vs. Non-uniform quantization, Symmetric vs. Asymmetric quantization

  > Linear Quantization: Integer-Arithmetic-Only Quantization, Sources of Quantization Error(clipping, rounding, scaling factor, zero point)

- [Vector Quantization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec05/summary02)

  > Vector Quantization(Deep compression: iterative pruning, K-means based quantization, Huffman encoding), Product Quantization

- [Post Training Quantization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec06/summary01)

  > Weight Quantiztion: Per-Tensor Activation Per-Channel Activation, Group Quantization(Per-Vector, MX), Weight Equalization, Adative Rounding

  > Activation Quantization: During training(EMA), Calibration(Min-Max, KL-divergence, Mean Squared Error)

  > Bias Correction, Zero-Shot Quantization(ZeroQ)

- [Quantization-Aware Training, Low bit-width quantization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec06/summary02)

  > Fake quantization, Straight-Through Estimator

  > Binary Quantization(Deterministic, Stochastic, XNOR-Net), Ternary Quantization

  ---

- [Neural Architecture Search: basic concepts & manually-designed neural networks](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec07/summary01)

  > input stem, stage, head
  
  > AlexNet, VGGNet, SqueezeNet(fire module), ResNet(bottleneck block, residual connection), ResNeXt(grouped convolution)
  
  > MobileNet(depthwise-separable convolution, width/resolution multiplier), MobileNetV2(inverted bottleneck block), ShuffleNet(channel shuffle), SENet(squeeze-and-excitation block), MobileNetV3(h-swish)

- [Neural Architecture Search: Search Space](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec07/summary02)

  > Search Space: Macro, Chain-Structured, Cell-based(NASNet), Hierarchical(Auto-DeepLab, NAS-FPN)

  > design search space: Cumulative Error Distribution, FLOPs distribution, zero-cost proxy

- [Neural Architecture Search: Performance Estimation & Hardware-Aware NAS](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec08)

  > Weight Inheritance, HyperNetwork, Weight Sharing(super-network, sub-network)

  > Performance Estimation Heuristics: Zen-NAS, GradSign

  ---

- [Knowledge Distillation](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec10/summary01)

  > Knowledge Distillation(distillation loss, softmax temperature)
  
  > What to Match?: intermediate weights, features(attention maps), sparsity pattern, relational information

  > Distillation Scheme: Offline Distillation, Online Distillation, Self-Distillation

- [Distillation for Applications](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec10/summary02)

  > Applications: Object Detection, Semantic Segmentation, GAN, NLP

  > Tiny Neural Network: NetAug

  ---

- [MCUNet](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec11)

  > MCUNetV1: TinyNAS, TinyEngine

  > MCUNetV2: MCUNetV2 architecture(MobileNetV2-RD), patch-based inference, joint automated search

### ⚙️ Efficient Training and System Support

- [Microcontroller, Loop Optimization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec17/summary01)

  > Memory Hierarchy of Microcontroller, Primary Memory Format(NCHW, NHWC, CHWN)

  > Parallel Computing Techniques: Loop Optimization(Unrolling, Reordering, Tiling)

- [Inference Optimization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec17/summary02)

  > Parallel Computing Techniques: SIMD Programming(CMSIS-NN)
  
  > Inference Optimization: Im2col, In-place, Choosing Data Layout(pointwise, depthwise), Winograd Convolution

  ---

### 🔧 Domain-Specific Optimizations

- [Transformer](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2023/lec12/summary01)

  > NLP Task(Discriminative, Generative), Pre-Transformer Era(RNN/LSTM, CNN)

  > Transformer: Tokenizer, Embedding, Multi-Head Attention(self-attention), Feed-Forward Network, Layer Normalization(Pre-Norm, Post-Norm), Positional Encoding

- [Transformer Design Variants](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2023/lec12/summary02)

  > Types of Transformer-based Models: Encoder-Decoder(T5), Encoder-only(BERT), Decoder-only(GPT)
  
  > Relative Positional Encoding(ALiBi, RoPE, interpolating RoPE), KV cache optimization(Multi-query Attention, Grouped-query Attention), Gated Linear Unit

  ---

- [LLM Quantization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2024/lec13/summary01)
  
  > Quantization Difficulty of LLMs, Bottleneck of edge LLM Inference(Memory-bounded, Memory footprint of Weights)
  
  > Weight-activation Quantization: SmoothQuant(Activation Smoothing)
  
  > Weight-only Quantization: AWQ(1% Salient Weights, Activation-aware Scaling)

- [Efficient System Support for LLM Quantization](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2024/lec13/summary02)
  
  > System for Edge: TinyChat(Hardware-aware Weight Packing, Kernel Fusion)

  > System for Cloud: Overhead in Quantized GEMM, QServe(SmoothAttention, Dequantization with Reg-Level Parallelism)

- [LLM Pruning & Sparsity](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2024/lec13/summary03)
  
  > Weight Sparsity: Wanda

  > Contextual Sparsity: Deja Vu, Mixture-of-Experts

  > Attention Sparsity: SpAtten, H2O

- [LLM Serving Systems](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2024/lec13/summary04)

  > Metrics for LLM Serving(TTFT, TPOT, Latency, Throughput),  Heuristics for Evaluation

  > PagedAttention(vLLM), FlashAttention, Speculative Decoding, Batching(Static, Dynamic, Continuous)

  ---

- [LLM Post Training](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2024/lec14/summary01)

  > Supervised Fine-Tuning, Reinforcement Learning from Human Feedback, Direct Preference Optimization

  > Parameter-Efficient Fine-Tuning: Additive(Adapter, Prompt/Prefix Tuning) Selective(BitFit), Reparameterized(LoRA)

  > PEFT Quantization: QLoRA, BitDelta

- [Prompt Engineering](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2024/lec14/summary02)

  > In-Context Learning: Zero-Shot Prompting, Few-Shot Prompting

  > Chain-of-Thought, Retrieval Augmented Generation

  ---

- [Vision Transformer](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2023/lec14/summary01)

  > Vision Transformer, High-Resolution Dense Prediction(Segment Anything)
  
  > Window Attention(Swin Transformer, FlatFormer), ReLU Linear Attention(EfficientViT, EfficientViT-SAM), Sparse Attention(SparseViT)

- [Efficient Video Understanding](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec19/summary01)

  > 2D CNNs for Video Understanding, 3D CNNs for Video Understanding(I3D), Temporal Shift Module(TSM)

  > Other Efficient Methods: Kernel Decomposition, Multi-Scale Modeling, Neural Architecture Search(X3D), Skipping Redundant Frames/Clips, Utilizing Spatial Redundancy

- [Generative Adversarial Networks (GANs)](https://github.com/erectbranch/MIT-Efficient-AI/tree/master/2022/lec19/summary02)

  > GANs(Generator, Discriminator), Conditional/Unconditional GANs, Difficulties in GANs

  > Compress Generator(GAN Compression), Dynamic Cost GANs(Anycost GANs), Data-Efficient GANs(Differentiable Augmenatation)
