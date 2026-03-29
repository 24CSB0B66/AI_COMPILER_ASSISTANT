import re, json, csv, argparse
from dataclasses import dataclass, field, asdict
from typing import List
from pathlib import Path
from datetime import datetime

TOKEN_PATTERNS = [
    ("KEYWORD(int)",    r'\bint\b'),
    ("KEYWORD(float)",  r'\bfloat\b'),
    ("KEYWORD(void)",   r'\bvoid\b'),
    ("KEYWORD(if)",     r'\bif\b'),
    ("KEYWORD(else)",   r'\belse\b'),
    ("KEYWORD(while)",  r'\bwhile\b'),
    ("KEYWORD(for)",    r'\bfor\b'),
    ("KEYWORD(return)", r'\breturn\b'),
    ("KEYWORD(char)",   r'\bchar\b'),
    ("IDENTIFIER",      r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ("FLOAT_LITERAL",   r'\b\d+\.\d+\b'),
    ("INT_LITERAL",     r'\b\d+\b'),
    ("EQ",              r'=='),
    ("NEQ",             r'!='),
    ("LTE",             r'<='),
    ("GTE",             r'>='),
    ("AND",             r'&&'),
    ("OR",              r'\|\|'),
    ("ASSIGN",          r'=(?!=)'),
    ("LT",              r'<'),
    ("GT",              r'>'),
    ("PLUS",            r'\+'),
    ("MINUS",           r'-'),
    ("MULTIPLY",        r'\*'),
    ("DIVIDE",          r'/'),
    ("MODULO",          r'%'),
    ("NOT",             r'!'),
    ("SEMICOLON",       r';'),
    ("COMMA",           r','),
    ("LPAREN",          r'\('),
    ("RPAREN",          r'\)'),
    ("LBRACE",          r'\{'),
    ("RBRACE",          r'\}'),
    ("LBRACKET",        r'\['),
    ("RBRACKET",        r'\]'),
    ("WHITESPACE",      r'[ \t\n\r]+'),
    ("UNKNOWN",         r'.'),
]
COMPILED_PATTERNS = [(n, re.compile(p, re.DOTALL)) for n,p in TOKEN_PATTERNS]

@dataclass
class Token:
    type: str; value: str; line: int; column: int
    def to_dict(self): return asdict(self)

@dataclass
class ErrorLabel:
    error_type: str; line: int; column: int; description: str; suggestion: str
    def to_dict(self): return asdict(self)

@dataclass
class CodeSample:
    sample_id: str; source_code: str; is_correct: bool
    errors: List[ErrorLabel] = field(default_factory=list)
    tokens: List[Token] = field(default_factory=list)
    token_sequence: List[str] = field(default_factory=list)
    category: str = "general"; difficulty: str = "beginner"
    def to_dict(self):
        return {"sample_id":self.sample_id,"source_code":self.source_code,
                "is_correct":self.is_correct,"errors":[e.to_dict() for e in self.errors],
                "tokens":[t.to_dict() for t in self.tokens],
                "token_sequence":self.token_sequence,
                "category":self.category,"difficulty":self.difficulty}

class Lexer:
    def tokenize(self, source):
        tokens=[]; pos=0; line=1; col=1
        while pos < len(source):
            match=None
            for tt, pat in COMPILED_PATTERNS:
                match=pat.match(source, pos)
                if match:
                    v=match.group(0)
                    if tt != "WHITESPACE":
                        tokens.append(Token(type=tt,value=v,line=line,column=col))
                    nl=v.count('\n')
                    if nl: line+=nl; col=len(v)-v.rfind('\n')
                    else: col+=len(v)
                    pos=match.end(); break
            if not match: pos+=1
        return tokens

class SyntaxErrorDetector:
    def detect(self, source, tokens):
        e=[]
        e.extend(self._semicolons(source,tokens))
        e.extend(self._braces(tokens))
        e.extend(self._parens(tokens))
        e.extend(self._assign_cond(tokens))
        return e

    def _semicolons(self, source, tokens):
        errors=[]; lines=source.split('\n')
        for i,line in enumerate(lines,1):
            s=line.strip()
            if (s and not s.endswith(';') and not s.endswith('{')
                and not s.endswith('}') and not s.endswith(')')
                and not s.startswith('//') and not s.startswith('#')
                and re.match(r'^[a-zA-Z_]',s)
                and re.search(r'[a-zA-Z0-9_\)"\']$',s)):
                if not re.match(r'^(if|else|while|for|void|int|float|char|struct)\s*[\(\{]',s):
                    if re.search(r'=\s*.+$',s) or re.search(r'\)\s*$',s):
                        errors.append(ErrorLabel("MISSING_SEMICOLON",i,len(line),
                            f"Statement on line {i} is missing a semicolon.",
                            f"Add ';' at end of line {i}: `{s};`"))
        return errors

    def _braces(self, tokens):
        stack=[]; errors=[]
        for tok in tokens:
            if tok.type=="LBRACE": stack.append(tok)
            elif tok.type=="RBRACE":
                if stack: stack.pop()
                else: errors.append(ErrorLabel("UNMATCHED_RBRACE",tok.line,tok.column,
                    f"Extra '}}' at line {tok.line} with no matching '{{'.",
                    "Remove the extra '}' or add a matching '{' earlier."))
        for tok in stack:
            errors.append(ErrorLabel("UNMATCHED_LBRACE",tok.line,tok.column,
                f"Opening '{{' at line {tok.line} has no matching '}}'.",
                "Add a closing '}' to close the block."))
        return errors

    def _parens(self, tokens):
        stack=[]; errors=[]
        for tok in tokens:
            if tok.type=="LPAREN": stack.append(tok)
            elif tok.type=="RPAREN":
                if stack: stack.pop()
                else: errors.append(ErrorLabel("UNMATCHED_RPAREN",tok.line,tok.column,
                    f"Extra ')' at line {tok.line} with no matching '('.",
                    "Remove the extra ')' or add a matching '(' earlier."))
        for tok in stack:
            errors.append(ErrorLabel("UNMATCHED_LPAREN",tok.line,tok.column,
                f"Opening '(' at line {tok.line} has no matching ')'.",
                "Add a closing ')' to match this parenthesis."))
        return errors

    def _assign_cond(self, tokens):
        errors=[]; i=0
        while i < len(tokens)-2:
            if tokens[i].type in ("KEYWORD(if)","KEYWORD(while)"):
                depth=0; j=i+1
                while j < len(tokens):
                    if tokens[j].type=="LPAREN": depth+=1
                    elif tokens[j].type=="RPAREN":
                        depth-=1
                        if depth==0: break
                    elif tokens[j].type=="ASSIGN" and depth>0:
                        errors.append(ErrorLabel("ASSIGNMENT_IN_CONDITION",
                            tokens[j].line,tokens[j].column,
                            f"'=' used inside condition at line {tokens[j].line}. Likely meant '=='.",
                            "Replace '=' with '==' to compare values."))
                    j+=1
            i+=1
        return errors

RAW_SAMPLES = [
    {"id":"C001","correct":True,"category":"variable_declaration","difficulty":"beginner",
     "code":"int x = 5;\nint y = 10;\nint sum = x + y;\n"},
    {"id":"C002","correct":True,"category":"if_statement","difficulty":"beginner",
     "code":"int a = 3;\nif (a > 0) {\n    return 1;\n} else {\n    return 0;\n}\n"},
    {"id":"C003","correct":True,"category":"while_loop","difficulty":"beginner",
     "code":"int i = 0;\nwhile (i < 10) {\n    i = i + 1;\n}\n"},
    {"id":"C004","correct":True,"category":"for_loop","difficulty":"intermediate",
     "code":"int sum = 0;\nfor (int i = 0; i < 5; i = i + 1) {\n    sum = sum + i;\n}\n"},
    {"id":"C005","correct":True,"category":"function","difficulty":"intermediate",
     "code":"int add(int a, int b) {\n    return a + b;\n}\n"},
    {"id":"C006","correct":True,"category":"nested_if","difficulty":"intermediate",
     "code":"int x = 5;\nif (x > 0) {\n    if (x < 10) {\n        int y = x * 2;\n    }\n}\n"},
    {"id":"C007","correct":True,"category":"array","difficulty":"intermediate",
     "code":"int arr[5];\narr[0] = 1;\narr[1] = 2;\nint val = arr[0] + arr[1];\n"},
    {"id":"C008","correct":True,"category":"recursion","difficulty":"advanced",
     "code":"int factorial(int n) {\n    if (n == 0) {\n        return 1;\n    }\n    return n * factorial(n - 1);\n}\n"},
    {"id":"C009","correct":True,"category":"for_if_else","difficulty":"intermediate",
     "code":"int main() {\n    int x;\n    int y;\n    x = 1;\n    for (int i = 0; i <= x; i = i + 1) {\n        x = x + 1;\n    }\n    if (x >= 1) {\n        y = x + 5;\n    } else {\n        y = x;\n    }\n    return 0;\n}\n"},
    {"id":"C010","correct":True,"category":"void_function","difficulty":"beginner",
     "code":"void greet(int x) {\n    int result = x + 1;\n    return;\n}\n"},
    {"id":"C011","correct":True,"category":"while_main","difficulty":"intermediate",
     "code":"int main() {\n    int x = 0;\n    while (x < 5) {\n        x = x + 1;\n    }\n    return x;\n}\n"},
    {"id":"C012","correct":True,"category":"nested_loop","difficulty":"advanced",
     "code":"int main() {\n    int sum = 0;\n    for (int i = 0; i < 3; i = i + 1) {\n        for (int j = 0; j < 3; j = j + 1) {\n            sum = sum + i + j;\n        }\n    }\n    return sum;\n}\n"},
    {"id":"C013","correct":True,"category":"multi_param_function","difficulty":"intermediate",
     "code":"int multiply(int a, int b, int c) {\n    int result = a * b * c;\n    return result;\n}\n"},
    {"id":"C014","correct":True,"category":"float_operations","difficulty":"intermediate",
     "code":"float average(float a, float b) {\n    float sum = a + b;\n    float avg = sum / 2;\n    return avg;\n}\n"},
    {"id":"C015","correct":True,"category":"array_loop","difficulty":"intermediate",
     "code":"int main() {\n    int arr[5];\n    int total = 0;\n    for (int i = 0; i < 5; i = i + 1) {\n        arr[i] = i * 2;\n        total = total + arr[i];\n    }\n    return total;\n}\n"},
    {"id":"C016","correct":True,"category":"logical_operators","difficulty":"intermediate",
     "code":"int check(int x, int y) {\n    if (x > 0 && y > 0) {\n        return 1;\n    }\n    if (x == 0 || y == 0) {\n        return 0;\n    }\n    return -1;\n}\n"},
    {"id":"C017","correct":True,"category":"nested_if_else","difficulty":"intermediate",
     "code":"int grade(int score) {\n    if (score >= 90) {\n        return 1;\n    } else if (score >= 70) {\n        return 2;\n    } else {\n        return 3;\n    }\n}\n"},
    {"id":"C018","correct":True,"category":"modulo_operator","difficulty":"beginner",
     "code":"int isEven(int n) {\n    int rem = n % 2;\n    if (rem == 0) {\n        return 1;\n    }\n    return 0;\n}\n"},
    {"id":"C019","correct":True,"category":"char_variable","difficulty":"beginner",
     "code":"int main() {\n    char c;\n    int x = 65;\n    return x;\n}\n"},
    {"id":"C020","correct":True,"category":"comparison_chain","difficulty":"intermediate",
     "code":"int clamp(int val, int lo, int hi) {\n    if (val < lo) {\n        return lo;\n    }\n    if (val > hi) {\n        return hi;\n    }\n    return val;\n}\n"},
    {"id":"C021","correct":True,"category":"while_counter","difficulty":"intermediate",
     "code":"int countDown(int n) {\n    int count = 0;\n    while (n > 0) {\n        n = n - 1;\n        count = count + 1;\n    }\n    return count;\n}\n"},
    {"id":"C022","correct":True,"category":"array_function","difficulty":"advanced",
     "code":"int sumArray(int arr[5]) {\n    int total = 0;\n    for (int i = 0; i < 5; i = i + 1) {\n        total = total + arr[i];\n    }\n    return total;\n}\n"},
    {"id":"C023","correct":True,"category":"void_main","difficulty":"beginner",
     "code":"void printNum(int n) {\n    int x = n * 2;\n    return;\n}\nint main() {\n    printNum(5);\n    return 0;\n}\n"},
    {"id":"C024","correct":True,"category":"fibonacci","difficulty":"advanced",
     "code":"int fib(int n) {\n    if (n == 0) {\n        return 0;\n    }\n    if (n == 1) {\n        return 1;\n    }\n    return fib(n - 1) + fib(n - 2);\n}\n"},
    {"id":"C025","correct":True,"category":"max_function","difficulty":"beginner",
     "code":"int maxOf(int a, int b) {\n    if (a > b) {\n        return a;\n    }\n    return b;\n}\n"},
    # ── INCORRECT SAMPLES E001-E025 ────────────────────────────────
    {"id":"E001","correct":False,"category":"missing_semicolon","difficulty":"beginner",
     "code":"int x = 5\nint y = 10;\nint sum = x + y;\n"},
    {"id":"E002","correct":False,"category":"unmatched_brace","difficulty":"beginner",
     "code":"if (x > 0) {\n    int y = 1;\n"},
    {"id":"E003","correct":False,"category":"assignment_in_condition","difficulty":"beginner",
     "code":"int x = 5;\nif (x = 0) {\n    return 1;\n}\n"},
    {"id":"E004","correct":False,"category":"unmatched_paren","difficulty":"beginner",
     "code":"int result = (a + b * c;\n"},
    {"id":"E005","correct":False,"category":"missing_semicolon","difficulty":"intermediate",
     "code":"int sum = 0\nfor (int i = 0; i < 5; i = i + 1) {\n    sum = sum + i;\n}\n"},
    {"id":"E006","correct":False,"category":"extra_brace","difficulty":"intermediate",
     "code":"int x = 5;\nif (x > 0) {\n    return x;\n}\n}\n"},
    {"id":"E007","correct":False,"category":"assignment_in_condition","difficulty":"intermediate",
     "code":"int i = 0;\nwhile (i = 10) {\n    i = i + 1;\n}\n"},
    {"id":"E008","correct":False,"category":"missing_semicolon","difficulty":"advanced",
     "code":"int factorial(int n) {\n    if (n == 0) {\n        return 1\n    }\n    return n * factorial(n - 1);\n}\n"},
    {"id":"E009","correct":False,"category":"multiple_errors","difficulty":"advanced",
     "code":"int add(int a, int b {\n    return a + b\n}\n"},
    {"id":"E010","correct":False,"category":"unmatched_brace","difficulty":"beginner",
     "code":"void greet() {\n    int x = 1;\n\n"},
    {"id":"E011","correct":False,"category":"missing_semicolon","difficulty":"intermediate",
     "code":"int main() {\n    int x;\n    int y;\n    x = 1;\n    for (int i = 0; i <= x; i = i + 1) {\n        x = x + 1;\n    }\n    if (x >= 1) {\n        y = x + 5;\n    } else {\n        y = x\n    }\n    return 0;\n}\n"},
    {"id":"E012","correct":False,"category":"assignment_in_condition","difficulty":"intermediate",
     "code":"int main() {\n    int x = 5;\n    if (x = 1) {\n        return x;\n    }\n    return 0;\n}\n"},
    {"id":"E013","correct":False,"category":"unmatched_brace","difficulty":"intermediate",
     "code":"int main() {\n    int x = 0;\n    while (x < 5) {\n        x = x + 1;\n    return x;\n}\n"},
    {"id":"E014","correct":False,"category":"missing_semicolon","difficulty":"beginner",
     "code":"int a = 10\nint b = 20;\nint c = a + b;\n"},
    {"id":"E015","correct":False,"category":"unmatched_paren","difficulty":"intermediate",
     "code":"int check(int x, int y {\n    return x + y;\n}\n"},
    {"id":"E016","correct":False,"category":"assignment_in_condition","difficulty":"beginner",
     "code":"int x = 3;\nint y = 7;\nif (x = y) {\n    return 1;\n}\n"},
    {"id":"E017","correct":False,"category":"missing_semicolon","difficulty":"intermediate",
     "code":"float avg(float a, float b) {\n    float sum = a + b\n    return sum / 2;\n}\n"},
    {"id":"E018","correct":False,"category":"unmatched_brace","difficulty":"advanced",
     "code":"int fib(int n) {\n    if (n == 0) {\n        return 0;\n    if (n == 1) {\n        return 1;\n    }\n    return fib(n-1) + fib(n-2);\n}\n"},
    {"id":"E019","correct":False,"category":"missing_semicolon","difficulty":"beginner",
     "code":"int main() {\n    int result = 0\n    return result;\n}\n"},
    {"id":"E020","correct":False,"category":"assignment_in_condition","difficulty":"advanced",
     "code":"int search(int arr[5], int key) {\n    for (int i = 0; i < 5; i = i + 1) {\n        if (arr[i] = key) {\n            return i;\n        }\n    }\n    return -1;\n}\n"},
    {"id":"E021","correct":False,"category":"unmatched_paren","difficulty":"intermediate",
     "code":"int sum = (a + b) * (c + d;\n"},
    {"id":"E022","correct":False,"category":"missing_semicolon","difficulty":"advanced",
     "code":"int power(int base, int exp) {\n    int result = 1\n    for (int i = 0; i < exp; i = i + 1) {\n        result = result * base;\n    }\n    return result;\n}\n"},
    {"id":"E023","correct":False,"category":"unmatched_brace","difficulty":"beginner",
     "code":"int main() {\n    int x = 5;\n    if (x > 0) {\n        return x;\n\n"},
    {"id":"E024","correct":False,"category":"assignment_in_condition","difficulty":"intermediate",
     "code":"int countPos(int n) {\n    int count = 0;\n    while (n = 0) {\n        n = n - 1;\n        count = count + 1;\n    }\n    return count;\n}\n"},
    {"id":"E025","correct":False,"category":"multiple_errors","difficulty":"advanced",
     "code":"int maxOf(int a, int b {\n    if (a = b) {\n        return a\n    }\n    return b;\n}\n"},
]

class DatasetBuilder:
    def __init__(self):
        self.lexer=Lexer(); self.detector=SyntaxErrorDetector()

    def build(self):
        samples=[]
        for raw in RAW_SAMPLES:
            tokens=self.lexer.tokenize(raw["code"])
            errors=[] if raw["correct"] else self.detector.detect(raw["code"],tokens)
            samples.append(CodeSample(
                sample_id=raw["id"],source_code=raw["code"],is_correct=raw["correct"],
                errors=errors,tokens=tokens,token_sequence=[t.type for t in tokens],
                category=raw["category"],difficulty=raw["difficulty"]))
        return samples

    def save_json(self, samples, path):
        data={"metadata":{"generated_at":datetime.now().isoformat(),
            "total_samples":len(samples),
            "correct_samples":sum(1 for s in samples if s.is_correct),
            "incorrect_samples":sum(1 for s in samples if not s.is_correct),
            "version":"2.0.0"},"samples":[s.to_dict() for s in samples]}
        with open(path,'w') as f: json.dump(data,f,indent=2)
        print(f"✓ JSON dataset saved → {path}")

    def save_csv(self, samples, path):
        with open(path,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(["sample_id","is_correct","category","difficulty",
                        "error_count","error_types","token_count","token_sequence","source_code"])
            for s in samples:
                w.writerow([s.sample_id,int(s.is_correct),s.category,s.difficulty,
                    len(s.errors),"|".join(e.error_type for e in s.errors),
                    len(s.tokens)," ".join(s.token_sequence),
                    s.source_code.replace('\n','\\n')])
        print(f"✓ CSV dataset saved  → {path}")

    def save_token_sequences(self, samples, path):
        with open(path,'w') as f:
            for s in samples:
                f.write(f"{'CORRECT' if s.is_correct else 'INCORRECT'}\t{' '.join(s.token_sequence)}\n")
        print(f"✓ Token sequences    → {path}")

    def print_stats(self, samples):
        total=len(samples); correct=sum(1 for s in samples if s.is_correct)
        errors={}; cats={}; diffs={}
        for s in samples:
            cats[s.category]=cats.get(s.category,0)+1
            diffs[s.difficulty]=diffs.get(s.difficulty,0)+1
            for e in s.errors: errors[e.error_type]=errors.get(e.error_type,0)+1
        print("\n"+"="*55)
        print("  DATASET STATISTICS")
        print("="*55)
        print(f"  Total samples    : {total}")
        print(f"  Correct          : {correct}  ({correct*100//total}%)")
        print(f"  Incorrect        : {total-correct}  ({(total-correct)*100//total}%)")
        print(f"\n  Error type distribution:")
        for k,v in sorted(errors.items(),key=lambda x:-x[1]): print(f"    {k:<35} {v}")
        print(f"\n  Category distribution:")
        for k,v in sorted(cats.items(),key=lambda x:-x[1]): print(f"    {k:<35} {v}")
        print(f"\n  Difficulty distribution:")
        for k,v in sorted(diffs.items(),key=lambda x:-x[1]): print(f"    {k:<35} {v}")
        print("="*55+"\n")

def main():
    ap=argparse.ArgumentParser(description="AI Compiler Dataset Builder v2.0")
    ap.add_argument('--build',action='store_true')
    ap.add_argument('--tokenize',action='store_true')
    ap.add_argument('--stats',action='store_true')
    ap.add_argument('--out',default='.')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    builder=DatasetBuilder(); samples=builder.build()
    if args.build or not any([args.build,args.tokenize,args.stats]):
        builder.save_json(samples,str(out/"dataset.json"))
        builder.save_csv(samples,str(out/"dataset.csv"))
        builder.save_token_sequences(samples,str(out/"token_sequences.txt"))
        builder.print_stats(samples)
    elif args.tokenize: builder.save_token_sequences(samples,str(out/"token_sequences.txt"))
    elif args.stats: builder.print_stats(samples)

if __name__=="__main__":
    main()
