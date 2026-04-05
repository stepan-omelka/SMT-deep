from smt_model.configuration_smt import SMTConfig

def build_model(config: SMTConfig, arch_type: str = "smt"):
    """
    Factory function to build either the original SMT architecture or the DeepSeek-OCR-2 architecture.
    """
    if arch_type == "smt":
        from smt_model.architectures.smt_arch import SMTModelForCausalLM
        return SMTModelForCausalLM(config)
    elif arch_type == "deepseek":
        from smt_model.architectures.deepseek_arch import DeepSeekOCR2Wrapper
        return DeepSeekOCR2Wrapper(config)
    elif arch_type == "qwen":
        from smt_model.architectures.qwen25_vl_arch import Qwen2_5_VLWrapper
        return Qwen2_5_VLWrapper(config)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}. Expected 'smt', 'deepseek', or 'qwen'.")
