# ZTL-27｜工程｜提示、路由与工具边界 · 思考闭合门控（</think> Gated）JSON 状态机词表掩码解码与业务值域二段校验
import json, math
from dataclasses import dataclass

VOCAB = ["<think>", "排查ERR-5042", "</think>", "```json", "{", '"action":', '"set_retry"', '"drop_all"', ',"timeout_ms":', "-200", "800", "}"]
FSM = {
    "S0": {"{"}, "S1": {'"action":'}, "S2": {'"set_retry"'},
    "S3": {',\"timeout_ms\":'}, "S4": {"-200", "800"}, "S5": {"}"},
}
NEXT = {"S0": "S1", "S1": "S2", "S2": "S3", "S3": "S4", "S4": "S5", "S5": "DONE"}

def step_decode(logits: list[float], in_think: bool, state: str) -> tuple[str, bool, str]:
    # 阶段一：<think> 内掩码全 0 自由推演；检出 </think> 后切入阶段二施加 -inf 硬掩码
    masked = logits if in_think else [z if t in FSM[state] else -math.inf for t, z in zip(VOCAB, logits)]
    m = max(masked)
    probs = [0.0 if z == -math.inf else math.exp(z - m) for z in masked]
    tok = VOCAB[max(range(len(VOCAB)), key=probs.__getitem__)]
    if in_think:
        return tok, (tok != "</think>"), state
    return tok, False, NEXT[state]

@dataclass
class RetryConfigModel:  # 等效服务端 Pydantic 业务值域二段校验器
    action: str
    timeout_ms: int
    def validate_bounds(self) -> str:
        if not (100 <= self.timeout_ms <= 5000):
            return f"VALUE_ERR: timeout_ms={self.timeout_ms} 越出 [100, 5000] 毫秒安全窗口"
        return "VALID_OK"

# 模拟轨迹：即便 ```json(4.2) 与非法枚举 "drop_all"(3.9) 分最高仍被屏蔽；但负数 -200 语法合法
trace_logits = [
    [5.0,0,0,0,0,0,0,0,0,0,0,0], [0,4.5,0,0,0,0,0,0,0,0,0,0], [0,0,4.8,0,0,0,0,0,0,0,0,0],
    [0,0,0,4.2,2.1,0,0,0,0,0,0,0], [0,0,0,0,0,3.1,0,0,0,0,0,0], [0,0,0,0,0,0,1.8,3.9,0,0,0,0],
    [0,0,0,0,0,0,0,0,3.0,0,0,0], [0,0,0,0,0,0,0,0,0,2.9,2.4,0], [0,0,0,0,0,0,0,0,0,0,0,3.5],
]
in_think, state, out = True, "S0", []
for lg in trace_logits:
    tok, in_think, state = step_decode(lg, in_think, state)
    out.append(tok)
raw_out = "".join(out)
payload = json.loads(raw_out.split("</think>", 1)[1])
print("1.门控解码输出:", raw_out)
print("2.二段值域核验:", RetryConfigModel(**payload).validate_bounds())
print("3.解码步数:", len(out), "| 思考段闭合:", not in_think, "| 终态:", state)
