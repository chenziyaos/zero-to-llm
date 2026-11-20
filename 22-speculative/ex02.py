# ZTL-22｜系统｜让草稿减少解码开销 · 串行步长寻优与 MTP 对 块扩散草稿 的 TPOT 对账
T_BASE_MS = 18.0

def expected_tokens(alpha: float, gamma: int) -> float:
    return float(gamma + 1) if alpha > 0.9999 else (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)

def serial_speedup(alpha: float, gamma: int, c: float, v_ratio: float = 1.04) -> float:
    return 1.0 if gamma == 0 else expected_tokens(alpha, gamma) / (gamma * c + v_ratio)

def solve_best_serial(alpha: float, c: float, max_g: int = 12) -> tuple:
    return max(((g, serial_speedup(alpha, g, c)) for g in range(0, max_g + 1)), key=lambda x: x[1])

for name, a, c, g in [("高温写作+7B重草稿", 0.44, 0.26, 4), ("通用对话+小草稿", 0.70, 0.14, 4), ("代码生成+EAGLE-3", 0.83, 0.04, 6)]:
    et, sf = expected_tokens(a, g), serial_speedup(a, g, c)
    gs, ss = solve_best_serial(a, c)
    print(f"{name:15s} | g={g}: E[t]={et:.2f}, S={sf:.2f}x (TPOT={T_BASE_MS/sf:5.2f}ms) | 最优 g*={gs:2d}->S*={ss:.2f}x")

tpot_mtp = (7 * 1.65 + 18.75) / 4.28
tpot_df2 = (1 * 8.20 + 18.80) / 4.80
print(f"原生串行 MTP (E[t]=4.28): TPOT={tpot_mtp:.2f}ms ({T_BASE_MS/tpot_mtp:.2f}x) | 并行块扩散 块扩散草稿 (E[t]=4.80): TPOT={tpot_df2:.2f}ms ({T_BASE_MS/tpot_df2:.2f}x)")
