import os
import ast

def clean_code():
    for root, dirs, files in os.walk('.'):
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()
                try:
                    parsed = ast.parse(source)
                    for node in ast.walk(parsed):
                        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
                            continue
                        if not len(node.body):
                            continue
                        if not isinstance(node.body[0], ast.Expr):
                            continue
                        if hasattr(node.body[0], 'value') and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                            node.body.pop(0)
                    cleaned = ast.unparse(parsed)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned)
                    print(f'Cleaned and formatted: {filepath}')
                except Exception as e:
                    print(f'Skipping {filepath} due to error: {e}')
if __name__ == '__main__':
    clean_code()