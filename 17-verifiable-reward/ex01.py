# ZTL-17｜后训练｜先验证奖励再训练 · 45 行纯标准库平衡花括号提取器、一次性令牌隔离代码沙箱与 T1～T8 红队反例全绿单测
import os, re, secrets, subprocess, sys, tempfile
from fractions import Fraction

def extract_boxes(s):
    out, i = [], 0
    while (idx := s.find(r"\boxed{", i)) != -1:
        depth, j = 1, idx + 7
        while j < len(s) and depth > 0:
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        if depth == 0: out.append(s[idx + 7 : j - 1].strip())
        i = j
    return out

def norm_frac(expr):
    expr = re.sub(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", r"\1/\2", expr.replace(" ", ""))
    if any(op in expr for op in ("=", "<", ">", "≈", "≠")) or not re.fullmatch(r"[+-]?\d+(?:/\d+)?", expr):
        return None
    return Fraction(expr)

def verify_math(resp, gold):
    m = re.fullmatch(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*", resp, flags=re.S)
    if not m or "<think>" in m.group(1) or "<answer>" in m.group(2): return 0.0
    think, ans = m.group(1), m.group(2)
    if len(think) >= 80 and len({think[k:k+4] for k in range(len(think) - 3)}) / len(think) < 0.15: return 0.0
    boxes = extract_boxes(ans)
    if len(boxes) != 1 or (pred := norm_frac(boxes[0])) is None: return 0.0
    return 1.0 if pred == norm_frac(gold) else 0.0

def verify_code(src, fn, cases, timeout=0.6):
    tok = secrets.token_hex(12)
    with tempfile.TemporaryDirectory() as tmp:
        tp = os.path.join(tmp, "ok.tok")
        code = (f"_open,_type=open,type\n{src}\nfor _a,_e in {cases!r}:\n"
                f"  _g={fn}(*_a)\n  if _type(_g) is not _type(_e) or _g!=_e: raise AssertionError\n"
                f"_open({tp!r},'w').write({tok!r})\n")
        try: r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired: return 0.0
        return 1.0 if (r.returncode == 0 and os.path.exists(tp) and open(tp).read() == tok) else 0.0

pos = [verify_math("<think>分子分母同约去2得二分之一</think><answer>\\boxed{\\frac{1}{2}}</answer>", "1/2"),
       verify_code("def solve(a, b): return a + b", "solve", [((2, 3), 5), ((-4, 9), 5)])]
adv = [verify_math("", "1/2"), verify_math("题干已知概率为 1/2，直接复抄 1/2", "1/2"),
       verify_math("<think>猜</think><answer>\\boxed{1}\\boxed{1/2}</answer>", "1/2"),
       verify_math("<think>伪装</think><answer>\\boxed{1/2=1/2}</answer>", "1/2"),
       verify_math("<think>" + "废话复读机" * 220 + "</think><answer>\\boxed{1/2}</answer>", "1/2"),
       verify_code("import os\ndef solve(a,b): os._exit(0)", "solve", [((2, 3), 5)]),
       verify_code("class F:\n def __eq__(s,o):return True\ndef solve(a,b):return F()", "solve", [((2, 3), 5)]),
       verify_code("def solve(a,b):\n while True: pass", "solve", [((2, 3), 5)], timeout=0.2)]
print("== 正例验证 =="); print("通过:", pos)
print("== 对抗反例验证 =="); print("T1~T8反例:", adv)
assert pos == [1.0, 1.0] and adv == [0.0] * 8; print("全部通过")
