# pretrained_weights 说明

`s3tokenizer.pt` 应为因果 S3 tokenizer 的导出权重（例如 Drive 中的 `s3tokenizer_export_epoch15.pt` 重命名放入），且与 Flow 训练时抽 token 所用为同一导出。

推理最少需要 5 个文件：

1. `s3tokenizer.pt`
2. `flow_torch_ddp/epoch_199_whole.pt`
3. `conf/cosyvoice_aishell_s3tok1024_25hz.yaml`
4. `CosyVoice-300M/campplus.onnx`
5. `CosyVoice-300M/hift.pt`

## A) Google Drive（核心 3 个文件）

- Drive: [s3_causal_flow folder](https://drive.google.com/drive/folders/1KwHVm4fNiKRTt-LqZrDkDSN9k9hEgC4B?usp=drive_link)

建议映射如下：

- `s3tokenizer_export_epoch15.pt` -> `pretrained_weights/s3tokenizer.pt`
- `epoch_199_whole.pt` -> `pretrained_weights/flow_torch_ddp/epoch_199_whole.pt`
- `cosyvoice_aishell_s3tok1024_25hz.yaml` -> `conf/cosyvoice_aishell_s3tok1024_25hz.yaml`

`epoch_199_whole.yaml` 为可选元信息文件，不是推理硬依赖。

## B) Hugging Face（补充 2 个文件）

- [campplus.onnx](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/campplus.onnx)
- [hift.pt](https://huggingface.co/FunAudioLLM/CosyVoice-300M/blob/main/hift.pt)

下载后放入：

```text
pretrained_weights/CosyVoice-300M/
  campplus.onnx
  hift.pt
```

> 可选：若要跑官方 preset（`official_cosyvoice1_50hz`），同目录还需 `flow.pt`、`cosyvoice.yaml`、`speech_tokenizer_v1.onnx`。
