import torch
from reinforcement_q_learning import DQN

def convert_model_to_torchscript(model, example_input, output_path):
    """
    将模型转换为 TorchScript 格式并保存为 .pt 文件。
    
    参数:
        model: 要转换的 PyTorch 模型。
        example_input: 用于追踪模型的示例输入。
        output_path: TorchScript 模型的保存路径。
    """
    # 将模型切换到评估模式
    model.eval()

    # 使用 torch.jit.trace 转换模型
    # 如果你的模型支持脚本化，也可以使用 torch.jit.script(model)
    traced_model = torch.jit.trace(model, example_input)

    # 保存为 TorchScript 格式
    traced_model.save(output_path)
    print(f"TorchScript 模型已保存到 {output_path}")

if __name__ == "__main__":
    # 示例：加载一个预训练的 ResNet18 模型
    model =torch.load('best_policy_net.pth', weights_only=False).to("cpu")

    # 示例输入（必须与模型的输入形状一致）
    example_input = torch.rand(1, 1, 4, 4)

    # 输出路径
    output_path = "best_policy_net.pt"

    # 转换并保存模型
    convert_model_to_torchscript(model, example_input, output_path)