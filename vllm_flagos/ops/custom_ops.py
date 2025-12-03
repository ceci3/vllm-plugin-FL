from vllm.model_executor.custom_op import CustomOp
from .layernorm import *
from .activation import *
from .rotary_embedding import *

    
def register_oot_ops():
    CustomOp.register_oot(_decorated_op_cls=FlagOSSiluAndMul, name="SiluAndMul")
    CustomOp.register_oot(_decorated_op_cls=FlagOSRMSNorm, name="RMSNorm")
    CustomOp.register_oot(_decorated_op_cls=FlagOSRotaryEmbedding, name="RotaryEmbedding")

