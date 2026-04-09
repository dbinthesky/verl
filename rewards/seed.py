#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Original Authors: Tim Henderson and Steve Johnson
#Email: tim.tadh@gmail.com, steve@steveasleep.com
#For licensing see the LICENSE file in the top level directory.

# This is a modified version of zss package.


import re
import pint  # NEW: For physical unit conversion and comparison
import collections
import numpy as np
import timeout_decorator
from sympy import simplify
from numpy import zeros,ones
from latex2sympy2_extended import *

from sympy import *
from sympy.core.function import AppliedUndef
from sympy.core.numbers import (
    Pi, Exp1,ImaginaryUnit,Infinity,NegativeInfinity,NaN,ComplexInfinity
)
from sympy.matrices import MatrixBase
from sympy.core.relational import Relational
from sympy import Derivative
from sympy.logic.boolalg import And, Or, Not
from sympy.simplify import *

class Node(object):


    def __init__(self, label, children=None):
        self.label = label
        self.children = children or list()
        

    @staticmethod
    def get_children(node):
        return node.children

    @staticmethod
    def get_label(node):
        return node.label

    def addkid(self, node, before=False):

        if before:  self.children.insert(0, node)
        else:   self.children.append(node)
        return self

    def get(self, label):

        if self.label == label: return self
        for c in self.children:
            if label in c: return c.get(label)

class AnnotatedTree(object):

    def __init__(self, root, get_children):
        self.get_children = get_children

        self.root = root
        self.nodes = list()  # a post-order enumeration of the nodes in the tree
        self.ids = list()    # a matching list of ids
        self.lmds = list()   # left most descendents of each nodes
        self.keyroots = None
            # the keyroots in the original paper
           

        stack = list()
        pstack = list()
        stack.append((root, collections.deque()))
        j = 0
        while len(stack) > 0:
            n, anc = stack.pop()
            nid = j
            for c in self.get_children(n):
                a = collections.deque(anc)
                a.appendleft(nid)
                stack.append((c, a))
            pstack.append(((n, nid), anc))
            j += 1
        lmds = dict()
        keyroots = dict()
        i = 0
        while len(pstack) > 0:
            (n, nid), anc = pstack.pop()
            self.nodes.append(n)
            self.ids.append(nid)
            if not self.get_children(n):
                lmd = i
                for a in anc:
                    if a not in lmds: lmds[a] = i
                    else: break
            else:
                try: lmd = lmds[nid]
                except:
                    import pdb
                    pdb.set_trace()
            self.lmds.append(lmd)
            keyroots[lmd] = i
            i += 1
        self.keyroots = sorted(keyroots.values())

    
def ext_distance(A, B, get_children, single_insert_cost,insert_cost,single_remove_cost, remove_cost, update_cost):
    '''Computes the extended tree edit distance between trees A and B with extended-zss algorithm
    Args:
        A(Node): Root node of tree 1
        B(Node): Root node of tree 2
        get_children(Func): the get_children method of tree
        single_insert_cost(Func): cost of inserting single node
        insert_cost(Func): cost of inserting a subtree
        update_cost(Func): cost of updating A to B


    Return:
        Distance(float):the tree editing distance
    '''
    A, B = AnnotatedTree(A, get_children), AnnotatedTree(B, get_children)
    size_a = len(A.nodes)
    size_b = len(B.nodes)
    treedists = zeros((size_a, size_b), float)
    fd=1000*ones((size_a+1,size_b+1),float)
    operations = [[[] for _ in range(size_b)] for _ in range(size_a)]


    def treedist(x, y):
        Al = A.lmds
        Bl = B.lmds
        An = A.nodes
        Bn = B.nodes

        m = size_a
        n = size_b

        fd[Al[x]][Bl[y]]=0
        for i in range(Al[x], x+1): 
            node = An[i]
            fd[i+1][Bl[y]] = fd[Al[i]][Bl[y]] + remove_cost(node)

        for j in range(Bl[y], y+1): 
            node = Bn[j]
            
            fd[Al[x]][j+1] = fd[Al[x]][Bl[j]] + insert_cost(node)

        for i in range(Al[x], x+1):
            for j in range(Bl[y], y+1):

                node1 = An[i]
                node2 = Bn[j]
                costs = [fd[i][j+1] + single_remove_cost(node1),
                             fd[i+1][j] + single_insert_cost(node2),
                             fd[Al[i]][j+1]+ remove_cost(node1),
                             fd[i+1][Bl[j]]+ insert_cost(node2)]
                m=min(costs)

                if Al[x] == Al[i] and Bl[y] == Bl[j]:
                    treedists[i][j]=min(m,fd[i][j]+update_cost(node1,node2))
                    fd[i+1][j+1]=treedists[i][j]
                else:
                    fd[i+1][j+1]=min(m,fd[Al[i]][Bl[j]]+treedists[i][j])


    for x in A.keyroots:
        for y in B.keyroots:
            treedist(x, y)

    return treedists[-1][-1]

def convert_caret_to_derivative(latex_str):
    # Match multiple consecutive ^ after variable names (2 or more)
    def repl(m):
        var = m.group(1)
        carets = m.group(2)
        n = len(carets)
        if n == 2:
            return f"{var}''"         # Second order uses double prime notation
        else:
            return f"{var}^{{({n})}}" # Higher orders use ^{(n)} notation
    pattern = r'([a-zA-Z]+)(\^{2,})'
    return re.sub(pattern, repl, latex_str)

def preprocess_special_superscripts(latex_str):
    # Define general variable pattern: variable name + optional subscript + optional existing superscript
    var_pattern = r'([a-zA-Z0-9_\\]+(?:_\{[^}]+\})?(?:\^\{[^}]+\})?)'

    # 1. Replace ^+ -> ^{+}
    latex_str = re.sub(fr'{var_pattern}\^\+', r'\1^{+}', latex_str)
    
    # 2. Replace ^- -> ^{-}
    latex_str = re.sub(fr'{var_pattern}\^\-', r'\1^{-}', latex_str)

    # 3. Replace ^* -> ^{star}
    latex_str = re.sub(fr'{var_pattern}\^\*', r'\1^{star}', latex_str)
    latex_str = re.sub(fr'{var_pattern}\^\{{(\\ast|\*)\}}', r'\1^{star}', latex_str)
    latex_str = re.sub(r'\^\{(\\ast|\*)\}', r'^{star}', latex_str)
    # 4. Replace invalid empty exponents with ^{prime}
    latex_str = re.sub(fr'{var_pattern}\^(?![\{{\\a-zA-Z0-9])', r'\1^{prime}', latex_str)

    return latex_str

def brackets_balanced(s: str) -> bool:
    """
    Check if the brackets in a LaTeX string are balanced
    Args:
        s(str): the input string
    Return:
        bool: True if the brackets are balanced, False otherwise
    """
    stack = []
    bracket_pairs = {')': '(', ']': '[', '}': '{'}  

    for char in s:
        if char in bracket_pairs.values():  
            stack.append(char)
        elif char in bracket_pairs:         
            if not stack or stack[-1] != bracket_pairs[char]:
                return False  
            stack.pop()        
    return len(stack) == 0  

def remove_non_ascii(text):
    """Remove non-ASCII characters from text"""
    return text.encode("ascii", errors="ignore").decode()

def extract_bracket_content(s: str, bracket_position: int) -> str:
    """Extract content within braces starting from given position"""
    start_idx=bracket_position

    stack = []
    content = []
    escaped = False
    brace_start=start_idx+1
    brace_depth = 0  
    for i in range(brace_start, len(s)):
        char = s[i]
        if escaped:
            content.append(char)
            escaped = False
            continue
        if char == '\\':
            escaped = True
            content.append(char)
            continue
        if char == '{':
            brace_depth += 1
            content.append(char)
        elif char == '}':
            if brace_depth == 0:
                return ''.join(content),i
            brace_depth -= 1
            content.append(char)
        else:
            content.append(char)

    return None,-1
def find_first_unescaped_brace(s: str) -> int:
    """Find the position of the first unescaped opening brace"""
    escaped = False
    for i, c in enumerate(s):
        if c == '\\' and not escaped:
            escaped = True
            continue
        if c == '{' and not escaped:
            return i
        escaped = False
    return -1

def extract_command(s: str, brace_pos: int) -> str | None:
    """extract the command name from a bracket"""
    i = brace_pos - 1
    parameter_mode=False
    while i >= 0:
        if not parameter_mode and s[i] in ('^','_'):
            return s[i]
        if not parameter_mode and not s[i] in (' ','\t',']','['):
            break
        if s[i]==']':
            parameter_mode=True
        if s[i]=='[' and parameter_mode:
            parameter_mode=False
        i -= 1
    
    # Start point
    if i < 0 or s[i] == '\\':
        return None
    
    # Extract command name
    command_end = i
    i -= 1
    while i >= 0 and s[i].isalpha():
        i -= 1
    if i<-1 or s[i]!='\\':
        return None
    return s[i+1:command_end+1]

def remove_command(s, command, keep_inside=False):
    """
    Removes all occurrences of a specified LaTeX-style command from a string using an iterative approach.

    This function is more robust and efficient than a recursive solution, avoiding recursion depth limits
    and excessive string copying.

    Args:
        s (str): The input string.
        command (str): The LaTeX-style command to remove (e.g., "\\textbf").
        keep_inside (bool, optional): If True, keeps the content inside the braces. Defaults to False.

    Returns:
        str: The modified string.

    Examples:
        >>> remove_command("This is \\textbf{bold text}.", "\\textbf")
        'This is '
        >>> remove_command("This is \\textbf{bold text}.", "\\textbf", keep_inside=True)
        'This is bold text.'
        >>> remove_command("Nested \\textbf{bold \\textit{italic text}} example.", "\\textbf", keep_inside=True)
        'Nested bold \\textit{italic text} example.'
        >>> remove_command("No braces \\here.", "\\here")
        'No braces .'
        >>> remove_command("Mismatched \\textbf{braces", "\\textbf")
        'Mismatched \\textbf{braces' # No replacement if brace is not closed
    """
    result_parts = []
    current_pos = 0
    while True:
        pos = s.find(command, current_pos)
        
        # If no more commands are found, end the loop
        if pos == -1:
            result_parts.append(s[current_pos:])
            break
            
        # 1. Add the part before the command
        result_parts.append(s[current_pos:pos])
        
        # Find the first character after the command, check if it's '{'
        brace_start_pos = pos + len(command)
        
        if brace_start_pos < len(s) and s[brace_start_pos] == '{':
            # Find the matching '}'
            level = 0
            brace_end_pos = -1
            for i in range(brace_start_pos, len(s)):
                if s[i] == '{':
                    level += 1
                elif s[i] == '}':
                    level -= 1
                    if level == 0:
                        brace_end_pos = i
                        break
            
            if brace_end_pos != -1:  # Successfully found matching bracket
                if keep_inside:        
                    # Keep the content inside the brackets
                    result_parts.append(s[brace_start_pos + 1 : brace_end_pos])
                # Update next search start position, skip the entire command and its content
                current_pos = brace_end_pos + 1
            else: # No matching bracket found, don't process
                # Add the command itself back, then start searching from after the command
                result_parts.append(s[pos:brace_start_pos + 1])
                current_pos = brace_start_pos + 1

        else: # No bracket after command, only remove the command itself
            current_pos = brace_start_pos

    return "".join(result_parts)

def convert_latex_fractions(latex_str):
    """Convert non-standard fractions to standard format"""
    pattern = r'\\frac((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))((?:\\[a-zA-Z]+|\d|[a-zA-Z]|{[^{}]*}))'
    
    def replacer(match):
        numerator, denominator = match.group(1), match.group(2)
        wrap_num = f'{{{numerator}}}' if not (numerator.startswith('{') and numerator.endswith('}')) else numerator
        wrap_den = f'{{{denominator}}}' if not (denominator.startswith('{') and denominator.endswith('}')) else denominator
        return fr'\frac{wrap_num}{wrap_den}'
    
    return re.sub(pattern, replacer, latex_str)

    
def get_first_brace_command(s: str) -> str | None:
    """ Find the position of the first unescaped opening brace and extract the command before it """
    brace_pos = find_first_unescaped_brace(s)
    if brace_pos == -1:
        return None
    return extract_command(s, brace_pos)
def remove_overall_brace(s: str) -> str:
    """Remove the outermost brace pair if it wraps the entire string"""
    pos=find_first_unescaped_brace(s)
    if pos==-1:
        return s,0
    command=get_first_brace_command(s)
    if not command:

        content,final=extract_bracket_content(s,pos)
        if final==len(s) or not '}' in s[final+1:]:
            return content,1
    return s,0

def exp_frac(s):
    """Add braces around exponentiated fractions"""

    def exp_frac_single(s):
        position=s.find("^\\frac")+1
        if position == 0:
            return s
        level=0
        cnt=0
        idx=position
        while idx<len(s):
            if s[idx]=='{':
                cnt+=1
            elif s[idx]=='}':
                cnt-=1
                if cnt==0:
                    level+=1
                    if level==2:
                        break
            idx+=1
        s1="".join([s[0:position],'{',s[position:idx],'}',s[idx:]])
        return s1
    s1=exp_frac_single(s)
    cnt=0
    while s1 != s and cnt<100:
        cnt+=1
        s=s1
        s1=exp_frac_single(s)
    return s

def find_all(s, sub_str, allow_overlap=True):
    """Find all occurrences of substring in string"""
    indexes = []
    start = 0
    step = 1 if allow_overlap else len(sub_str)
    cnt=0
    while True and cnt<100:
        pos = s.find(sub_str, start)
        if pos == -1:
            break
        indexes.append(pos)
        start = pos + step 
        cnt+=1
    return indexes

def bar_inside_vec(s):
    """Handle bar notation inside vector commands"""
    indices=find_all(s,"\\vec{")
    if not indices:
        return s
    for i in range(len(indices)):
        position=find_all(s,"\\vec{")[i]
        idx=position+4
        idx2=idx
        level=0
        while idx2<len(s):
            if s[idx2]=='{':
                level+=1
            if s[idx2]=='}':
                level-=1
                if level==0:
                    break
            idx2+=1
    
        s1=s[idx+1:idx2]

        s1=remove_command(s1,"\\bar",keep_inside=True)
        s2= "".join([s[0:idx+1],s1,s[idx2:]])
        s=s2
    return s

def vec_lower_idx(input_str):
    """ 
    Args:
        input_str (str): Original string
    
    Returns:
        str: Converted string
    """
    pattern = r'\\vec\{([^{}]+)_{([^{}]+)}\}'
    replacement = r'\\vec{\1}_{\2}'
    return re.sub(pattern, replacement, input_str)

def convert_vec_syntax(text):
    """
    Converts LaTeX vector syntax to a standardized form.

    This function processes a given text string and ensures that LaTeX vector 
    notations are consistently formatted. Specifically, it transforms instances 
    of `\vec xxx` into `\vec{xxx}`. The function handles cases where the vector 
    notation is applied to single characters, Greek letters, or LaTeX commands.

    Args:
        text (str): The input string containing LaTeX code to be processed.

    Returns:
        str: The processed string with standardized vector syntax.

    Examples:
        >>> convert_vec_syntax(r"\vec x + \vec\alpha + \vec\Gamma")
        '\\vec{x} + \\vec{\\alpha} + \\vec{\\Gamma}'
    """
    
    pattern = r'\\vec(\s*)(\\?[a-zA-Zα-ωΑ-Ω]+)'
    replacement = r'\\vec{\2}'
    return re.sub(pattern, replacement, text)

def remove_outer_braces(tex_str):
    """
    Convert {base}_{subscript} to base_{subscript} 
    Example
    {a}_{xyz} → a_{xyz}
    {\theta}_{0} → \theta_{0}
    """

    pattern = r'\{(\\(?:[a-zA-Z]+|.)|[^{}])+\}_\{([^}]+)\}'
    return re.sub(pattern, r'\1_{\2}', tex_str)

def extract_last_equal_content(s: str, strip_whitespace: bool = True) -> str:
    """
    Extract the content after the last occurrence of specific mathematical comparison or assignment operators.

    :param strip_whitespace: If True, removes leading and trailing whitespace from the extracted content. Defaults to True.
    (e.g., '=', '\\approx', '\\ge', '\\le', etc.) within the input string `s`. It then extracts 
    and returns the content that follows the operator. If no operator is found, the entire string 
    is returned. Optionally, leading and trailing whitespace can be stripped from the extracted content.

    Args:
        s (str): The input string to process.
        strip_whitespace (bool): Whether to strip leading and trailing whitespace from the extracted content. Defaults to True.

    Returns:
        str: The content after the last matching operator, or the entire string if no operator is found.
    """
    comparison_operators=('\\approx','\\ge','\\le','\\geq','\\leq','=')
#'\\approx','\\ge','\\le','\\geq','\\leq','<','>',
    content=s
    for sign in comparison_operators:
        if sign in s:
            rfind_index = s.rfind(sign)
            if s[rfind_index:rfind_index+5]=="\\left" and sign=='\\le':
                continue
            if rfind_index != -1:
                content = s[rfind_index + 1:]
                if content =="0":
                    print("")
    if strip_whitespace:
        return content.strip()
    return content

def first_pre_process(s,t,extract_box=True):
    """
    Perform the first stage of LaTeX string preprocessing.

    if not brackets_balanced(s):
        raise ValueError("The input string has unbalanced brackets. Please check the LaTeX expression.")
    equality or comparison operator.

    Args:
        s (str): The input LaTeX string to preprocess.
        extract_box (bool): If True, extracts the content inside a '\\boxed' command. Defaults to True.

    Returns:
        str: The preprocessed LaTeX string.
    """
    #s=remove_non_ascii(s)
    s=s.replace('\\{','(') 
    s=s.replace('\\}',')')

    if t == "Expression" or t == "Equation":
        s = s.replace('\\approx', '=')
        
    if not brackets_balanced(s):
        return s
    if extract_box:
        boxed_content=remove_command(s,'\\boxed',keep_inside=True)
    else:
        boxed_content=s
    exist_overall_brace=True
    cnt=0
    while exist_overall_brace and cnt<10:
        boxed_content,exist_overall_brace=remove_overall_brace(boxed_content)
        cnt+=1

    if '\\quad' in boxed_content:
        boxed_content = boxed_content.split('\\quad')[0]

    if '\\qquad' in boxed_content:
        boxed_content = boxed_content.split('\\qquad')[0]

        boxed_content = boxed_content.strip(' \\') 

    if t == "Equation":
        last_equal_content = boxed_content
    else:
        last_equal_content = extract_last_equal_content(boxed_content)


    # last_equal_content=extract_last_equal_content(boxed_content)

    exist_overall_brace=True
    cnt=0
    while exist_overall_brace and cnt<10:
        last_equal_content,exist_overall_brace=remove_overall_brace(last_equal_content)
        cnt+=1
    return last_equal_content

def remove_text_from_latex(expr: str) -> str:
    """Replace Chinese characters with '1' characters"""
    def repl(match):
        length = len(match.group())
        return '1' * length
    return re.sub(r'[\u4e00-\u9fa5]+', repl, expr)

def extract_bracket_subscript_pairs(expr):
    """Extract bracket-subscript pairs from expression"""
    matches = []
    stack = []
    i = 0
    n = len(expr)

    while i < n:
        if expr[i] in '({[':
            stack.append((i, expr[i]))
        elif expr[i] in ')}]':
            if not stack:
                i += 1
                continue
            start, open_br = stack.pop()
            close_br = expr[i]
            if (open_br, close_br) not in [('(', ')'), ('[', ']'), ('{', '}')]:
                i += 1
                continue

            j = i + 1
            if j < n and expr[j] == '_':
                k = j + 1
                if k < n and expr[k] == '{':
                    k += 1
                    while k < n and expr[k] != '}':
                        k += 1
                    k += 1
                else:
                    k += 1
                matches.append((start, k, expr[start:k]))
        i += 1
    return matches

def add_number_to_bracket_subscripts(expr):
    """Add numbering to bracket subscripts"""
    matches = extract_bracket_subscript_pairs(expr)
    if not matches:
        return expr

    matches.sort(reverse=True)
    counter = 1
    for start, end, content in matches:
        new_content = re.sub(r'(_)', f'{counter}\\1', content, count=1)
        expr = expr[:start] + new_content + expr[end:]
        counter += 1
    return expr

def insert_multiplication_symbols(expr):
    """
    Automatically insert \cdot in LaTeX expressions where needed, handling implicit multiplication cases.
    Example: \frac{1}{2}\bar{E}1_a^i → \frac{1}{2} \cdot \bar{E} \cdot 1_a^i
    """

    # Add \cdot after \frac{...}{...} if directly followed by variables or functions
    expr = re.sub(r'(\\frac\{[^}]+\}\{[^}]+\})(?=\\[a-zA-Z]|[a-zA-Z0-9])', r'\1 \\cdot ', expr)

    # Insert \cdot between a symbol (like \bar{E}) and another variable
    expr = re.sub(r'(\})((\d|[a-zA-Z])_?[a-zA-Z]?\^?[a-zA-Z]?)', r'\1 \\cdot \2', expr)

    return expr

def remove_all_text_commands(latex_str):
    """
    Remove all \text{...} commands and their content from LaTeX.
    Args:
        latex_str (str): Input LaTeX string
    Returns:
        str: String after removing \text{...}
    """
    pattern = r'\\text\{[^{}]*\}'
    return re.sub(pattern, '1', latex_str)
def convert_general_exp_format(latex_str):
    # Match patterns like x^{*2}, f(x)^{*3}, \alpha^{*4}, etc.
    pattern = r"([a-zA-Z\\]+|\([^)]+\)|\{[^}]+\})\^\{\*(\d+)\}"
    
    # Convert to (base^*)^n format
    return re.sub(pattern, r"(\1^*)^\2", latex_str)
def modify_latex_expression(expr: str) -> str:
    # Replace V_{CKM}^{ji*} with V_{CKM}^ji^*
    expr = re.sub(r'V_\{CKM\}\^\{([^\}]*?)\*\}', r'V_{CKM}^\1', expr)

    # Remove + appearing before \text
    expr = re.sub(r'\+\s*(\\text)', r'\1', expr)

    return expr

def wrap_single_subscripts(s: str) -> str:
    """
    Convert subscripts like xxx_Y or xxx_y to xxx_{Y}/xxx_{y}.
    
    - Only handle single English letters
    - If subscript is already _{...} or followed by \command, don't modify
    """
    # Negative lookahead (?![{\\]): exclude _{ already bracketed and _\command cases
    pattern = re.compile(r'_(?![{\\])([A-Za-z])')
    return pattern.sub(r'_{\1}', s)

def replace_hc_text(s: str) -> str:
    """
    Replace \text{h.c.} (case and space insensitive) with h_c,
    keep other \text{...} unchanged.
    """
    pattern = re.compile(r'\\text\s*{([^{}]*)}')

    def repl(m):
        content = m.group(1).strip()
        norm = content.lower().replace(' ', '')
        if norm in ('h.c.', 'h.c'):
            return 'h_c'
        return m.group(0)

    return pattern.sub(repl, s)

def standardize_dE_notation(s: str) -> str:
    s = re.sub(r'd\*([A-Z])_({?[a-zA-Z0-9]+}?)', r'd{\1}_\2', s)
    return s

def replace_arrow_expression(s: str) -> str:
    """
    Replace W(i arrow f) with W(iRf), i.e., change 'i arrow f' to 'iRf' in parentheses.
    """
    return re.sub(r'W\(\s*(\w+)\s+arrow\s+(\w+)\s*\)', r'W(\1R\2)', s)

def preprocess_feynman_slash(latex_str: str) -> str:
    """
    Converts Feynman slash notation like \not{k} into a plain variable `kslash`.
    This helps latex2sympy to parse specialized physics notations.
    Example: \not{k}_0 -> kslash_0
    """
    pattern = r'\\not\{([^{}]+)\}'
    
    replacement = r'\\bar{\1slash}'
    
    return re.sub(pattern, replacement, latex_str)

def fix_subscript_on_parentheses(s: str) -> str:

    # Match pattern: (content)_{subscript}
    pattern = r'\(([^)]+)\)_\{([^}]+)\}'
    
    # Replacement rule: keep only "content" and "subscript", remove outer ()
    replacement = r'\1_{\2}'
    
    return re.sub(pattern, replacement, s)


def reorder_super_sub(latex_str: str) -> str:
    """
    Reorder base^{super}_{sub} form to base_{sub}^{super}.
    Example: M^{-1}_{j_1 i_1} -> M_{j_1 i_1}^{-1}
    This function can handle single letters, multiple letters, and LaTeX commands as base symbols.
    """
    # Pattern: (base symbol)(superscript)(subscript)
    # Base symbol: one or more letters, possibly starting with backslash
    # Superscript: ^{...}
    # Subscript: _{...} 
    pattern = r'([a-zA-Z\\]+)(\^\{[^}]+\})(_\{[^}]+\})'
    replacement = r'\1\3\2'
    
    # Continuously apply replacement until the string no longer changes
    # This is a safer approach for handling more complex cases (though not needed in this example)
    while True:
        new_str = re.sub(pattern, replacement, latex_str)
        if new_str == latex_str:
            break
        latex_str = new_str
        
    return latex_str

def second_pre_process(s):
    """
    Perform the second stage of LaTeX string preprocessing.

    This function removes or modifies specific LaTeX commands and content to standardize
    the input string for further processing. It handles commands like '\\text', '\\mathbf',
    and '\\mathrm', removes unnecessary content, and applies transformations such as
    converting fractions and vector syntax.

    Args:
        s (str): The input LaTeX string to preprocess.

    Returns:
        str: The preprocessed LaTeX string.
    """

    s = reorder_super_sub(s)

    kill_commands=[
        '\\begin',
        '\\end'
    ]
    remove_commands=[
        '\\text',
        '\\mathbf',
        '\\mathrm',
        '\\mathscr',
        '\\mathcal',
        '\\mathfrak',
        '\\pmb',
        '\\hat',
        '\\overline',
        '\\boldsymbol',
        '\\mathbb',
    ]


    remove_content=[
        '\\,','$',',','`','latex','\\left','\\right','\\text','\\mathrm','\\Bigr','\\Bigl','\n','\\]','\\[',
        '\\Big','\\bigl','\\bigr','\\biggl','\\biggr','\\displaystyle','\\boldsymbol','\\infty'
    ]
    replace_content=[
        ('\\operatorname{asin}','\\asin'),
        ('\\operatorname{sech}','\\sech'),
        ('\\operatorname{acos}','\\acos'),
        ('\\operatorname{sinh}','\\sinh'),
        ('\\operatorname{rot}','\\bar{rot}'),
        ('\\dfrac','\\frac'),
        ('\\tfrac','\\frac'),
        ('\\Exp','\\exp'),
        ('\\gg','>'),
        ('\\ll','<'),
        ('\\times','\\bar{times}'),
        ('\\dagger','\\bar{dagger}'),
        ('\\operatorname{dim}','\\bar{dim}'),
        ('\\overleftarrow','\\bar{overleftarrow}'),
        ('\;',' '),
        (';','\\bar{CD}'),
        ('\\partial','\\bar{partial}'),
        ('\\perp','\\bar{perp}'),
        ('\\parallel','\\bar{parallel}'),
        ('\\|','\\bar{parallel}'),
        ('\\epsilon','\\varepsilon'),
        ('\\varOmega','\\Omega'),
        ('I','\\bar{I}'),
        ('_e','_{e}'),
        ('e_','\\bar{e}_'),
        ('E_','\\bar{E}_'),
        ('\\pm','+'),
        ('\\mp','-'),
        ('{+}','{p}'),
        ("{-}",'{m}'),
        ("_+",'_p'),
        ('_-',"_m"),
        # ('\\infty', 'oo')
    ]

    # More precise handling of single quotes: distinguish derivatives and physics symbols
    # Handle function derivatives: f'(x) -> f^{prime}(x)
    s = re.sub(r'([a-zA-Z]+)\'(?=\()', r'\1^{prime}', s)
    s = re.sub(r'([a-zA-Z]+)\'(?=\s|$|[^a-zA-Z(])', r'\1^{prime}', s)
    # Handle single quotes in braces: {k}' -> {k}^{prime}
    s = re.sub(r'(\{[a-zA-Z]+\})\'', r'\1^{prime}', s)
    s = re.sub(r'·', '', s)
    # s = s.replace(r'\dagger', 'dagger')
    # s = re.sub(r'\|(.+?)\\rangle', r'\1', s)
    s = s.replace(r'\operatorname{Im}', 'Im')
    # Remove angle brackets from Dirac symbols or inner product symbols
    s = re.sub(r'\\langle\s*(.+?)\s*\\rangle', r'{\1}', s)
    s = re.sub(r'\|\s*(.+?)\s*\\rangle', r'\1', s)
    s = s.replace(r'\sim', 'Symbol("sim")')
    s = re.sub(r'\\bar\{([^{}]+)\}', r'\1', s)
    s = replace_hc_text(s)
    s = convert_general_exp_format(s)
    s = convert_caret_to_derivative(s)
    s = preprocess_special_superscripts(s)   
    s = wrap_single_subscripts(s)
    s = modify_latex_expression(s)
    s = remove_all_text_commands(s)
    s = fix_subscript_on_parentheses(s)

    # s=remove_outer_braces(s)
    # Special case: protect differential forms, avoid E_ replacement affecting dE_{k}
    # Handle normal form: dE_{k}
    s = re.sub(r'\bd([A-Z])_', r'd\1UNDERSCORE', s)
    # Handle mathbf form: d\mathbf{E}_{k}
    s = re.sub(r'\bd\\mathbf\{([A-Z])\}_', r'd\\mathbf{\1}UNDERSCORE', s)

    s = re.sub(r'\\ddot\{([^}]+)\}', r'\1_{ddot}', s)
    s = re.sub(r'\\ddot([A-Za-z]+)', r'\1_{ddot}', s)
    # Similarly handle \dot
    s = re.sub(r'\\dot\{([^}]+)\}', r'\1_{dot}', s)
    s = re.sub(r'\\dot([A-Za-z]+)', r'\1_{dot}', s)
    # If the string contains matrix environment keywords, skip kill_commands processing
    if not ('\\begin{pmatrix}' in s or '\\end{pmatrix}' in s or
            '\\begin{bmatrix}' in s or '\\end{bmatrix}' in s or
            '\\begin{matrix}' in s or '\\end{matrix}' in s or
            '\\begin{vmatrix}' in s or '\\end{vmatrix}' in s or
            '\\begin{Vmatrix}' in s or '\\end{Vmatrix}' in s):

        for command in kill_commands:
            s=remove_command(s,command,keep_inside=False)
    for command in remove_commands:
        s=remove_command(s,command,keep_inside=True)
    for content in remove_content:
        s=s.replace(content,'')
    for content in replace_content:
        s=s.replace(content[0],content[1])
    # Restore protected differential forms and add multiplication signs for latex2sympy recognition
    if '\\lim' in s:
        s = s.replace(r'arrow', r'\rightarrow')
    else:
        s = re.sub(r'\barrow\b', r'\\bar{arrow}', s)
    s = re.sub(r'd([A-Z])UNDERSCORE', r'd*\1_', s)
    s = re.sub(r'd\\mathbf\{([A-Z])\}UNDERSCORE', r'd*\\mathbf{\1}_', s)
    s = preprocess_feynman_slash(s)
    s= convert_latex_fractions(s)
    s = standardize_dE_notation(s)
    # s = replace_arrow_expression(s)
    s=bar_inside_vec(s)
    s=vec_lower_idx(s)
    s=convert_vec_syntax(s)
    s=exp_frac(s)
    if s and s[-1] == '.':
        s = s[:-1]
    s = s.replace(r'\varkappa', r'\kappa')
    # First replace derivative forms to avoid parsing errors
    s = replace_derivative_frac_preserve_frac(s)
    s = remove_text_from_latex(s)
    s = add_parentheses_to_d(s)
    s = add_number_to_bracket_subscripts(s)
    s = insert_multiplication_symbols(s)
    s = s.replace('Å', 'A')
    return s

def add_parentheses_to_d(expr):
    """
    Pattern: match a 'd', but ensure it's not preceded by \frac{
    (?<!\\frac{) is "negative lookbehind assertion", requiring that the match position cannot be preceded by "\frac{"
    The \ in (?<!...) needs to be escaped, so it's (?<!\\frac{)
    """
    pattern = r'(?<!\\frac{)d(\\[A-Za-z0-9_]+)'
    
    # Replacement rule unchanged
    return re.sub(pattern, r'd(\1)', expr)



class MyConfig:
    
    interpret_as_mixed_fractions: bool = False
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False

    """
    Args:
        interpret_as_mixed_fractions (bool): Whether to interpert 2 \frac{1}{2} as 2/2 or 2 + 1/2
        interpret_simple_eq_as_assignment (bool): Whether to interpret simple equations as assignments k=1 -> 1
        interpret_contains_as_eq (bool): Whether to interpret contains as equality x \\in {1,2,3} -> x = {1,2,3}
        lowercase_symbols (bool): Whether to lowercase all symbols
    """
class MyNormalization:
    """Configuration for latex normalization.
    
    Each field controls a group of related normalizations:
    - basic_latex: Basic latex command replacements (mathrm, displaystyle, etc.)
    - units: Remove units and their variations
    - malformed_operators: Fix malformed operators (sqrt, frac, etc.)
    - nits: Small formatting fixes (spaces, dots, etc.)
    - boxed: Extract content from boxed environments
    - equations: Handle equation splitting and approximations (deprecated)
    """

    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = True
    nits: bool = True
    boxed = "all"
    equations: bool = False


def replace_derivative_frac_preserve_frac(expr: str) -> str:
    """
    Convert d<var> in \frac{d<var1>}{d<var2>} to symbol names, preserve \frac structure,
    preserve underscores _.
    """
    pattern = r'''
        \\frac\{
            d
            (\\?[a-zA-Z]+)
            (_\{?[a-zA-Z0-9]+\}?)?
        \}\{
            d
            (\\?[a-zA-Z]+)
            (_\{?[a-zA-Z0-9]+\}?)?
        \}
    '''

    def clean(s):
        return s.replace('\\', '').replace('{', '').replace('}', '')

    def repl(m):
        var1 = clean(m.group(1))
        sub1 = clean(m.group(2) or '')
        var2 = clean(m.group(3))
        sub2 = clean(m.group(4) or '')

        return f'\\frac{{D{var1}{sub1}}}{{D{var2}{sub2}}}'

    return re.sub(pattern, repl, expr, flags=re.VERBOSE)

@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def master_convert_with_timeout(s, t):
    """Master convert with timeout protection"""
    s = re.sub(r'~', '', s)
    preprocessed_stage1 = first_pre_process(s, t)
    preprocessed_stage2 = second_pre_process(preprocessed_stage1)
    Sym = latex2sympy(preprocessed_stage2, normalization_config=MyNormalization(), conversion_config=MyConfig())
    return Sym

def master_convert(s,t):
    """
    The only function needed to convert a LaTeX string into a SymPy expression.

    Args:
        s (str): The input LaTeX string. It should be a valid LaTeX mathematical expression, 
                 such as equations, fractions, or symbols, and must have balanced brackets.

    Returns:
        Sym (Sympy Expression): A SymPy expression representing the mathematical content of the input string.
                                The returned object can be used for symbolic computation, simplification, 
                                or evaluation using SymPy's functionality.

    Example:
        >>> master_convert("\\frac{1}{2} + x")
        1/2 + x
    """
    try:    
        return master_convert_with_timeout(s, t)
    except TimeoutError:
        print(f"  -> master_convert timeout for LaTeX: {s[:100]}...")
        return None
    except Exception as e:
        print(f"  -> master_convert error: {e}")
        return None



"""
There are four main categories:

Constants: such as integers, decimals, or mathematical constants like π and e.
Variables: letters like x, y, z, or specified terms in problems (e.g., ħ, c, G).
Functions: sine, cosine, exponential, logarithm, etc.
Operators: basic binary operations including addition, multiplication, and exponentiation.
"""
# The costs can be modified if you think their values are different
insert_cost={"number":1,"symbol":1,"operator":1,"function":1,"matrix":1,"relation":1}
delete_cost={"number":1,"symbol":1,"operator":1,"function":1,"matrix":1,"relation":1}
update_cost={"number":1,"symbol":1,"operator":1,"function":1,"matrix":1,"relation":1}

change_type_cost=1 #the cost of an update between different types,can be set to higher

bar_size=5 # the minimum size of triggering cluster discount
discount_slope=0.6 #discount

simplify_time_limit=30 #set the time limit of simplify
equals_time_limit=10 #set the time limit of equals

def update_func(x,y):
    
    if x.label==y.label:
        return 0
    
    elif x.label.split("_")[0]==y.label.split("_")[0]:
        return update_cost[x.label.split("_")[0]]
    return change_type_cost
def remove_func(x):
    return delete_cost[x.label.split("_")[0]]

def remove_tree_func(x):
    if not x.children:
        return remove_func(x)
    s=calc_tree_size(x)
    return min(s,discount_slope*(s-bar_size)+bar_size)

def insert_func(x):
    return insert_cost[x.label.split("_")[0]]
def insert_tree_func(x):
    return remove_tree_func(x)

def calc_tree_size(node):
    """
    Calculate the size of a subtree based on its total insertion cost.
    
    The function computes the size of a subtree by summing up the insertion 
    costs of the current node and all its descendant nodes. If the subtree 
    size has already been calculated and stored in `node.subtree_size`, it 
    returns the cached value to avoid redundant computation.
    
    Args:
        node (Node): The root node of the subtree for which the size is to 
                     be calculated
    Returns:
        int: The total size of the subtree, calculated as the sum of the 
             insertion costs of the current node and all its descendants.
    Notes:
        - The `insert_cost` dictionary is assumed to be globally defined 
          and maps node labels to their respective insertion costs.
        - The function modifies the `subtree_size` attribute of the input 
          node to store the calculated subtree size for future use.
    """
    """The size of a subtree equals to its total insertion cost"""
    
    total = insert_cost[node.label.split("_")[0]]
    
    if node.children and node.subtree_size !=0:

        return node.subtree_size
    
    for child in node.children:
        total += calc_tree_size(child)
    
    node.subtree_size=total

    return total
"""
Scoring function from relative distance
"""
def score_calc(tree_dist,tree_size):

    if tree_dist==0.:
        return 100
    return max(0,100*discount_slope-100*tree_dist/tree_size)

def numeric_score_calc(student_answer_exp, ground_truth_exp):
    """
    Specialized scoring function for numeric types
    Scores based on combined criteria of absolute and relative errors with configurable thresholds
    Features
    - Multi-tier scoring: 100pts (0.5% tolerance), 90pts (1%), 80pts (2%)
    - Sign consistency checking to catch conceptual errors
    - Special handling for zero values
    - Graceful fallback to tree-based scoring on conversion failures
    """
    #  Parameter Setting Section (Adjust scoring strictness)

    # 100-point standard (strictest)
    RelTol_100_strict = 0.01  # 1%
    
    # 90-point standard (moderately strict)
    RelTol_90 = 0.02   # 2%
    
    # 80-point standard (more lenient)
    RelTol_80 = 0.04   # 4%
    
    try:
        # If ground_truth_exp is an equation, extract the right-hand side value
        if hasattr(ground_truth_exp, 'rhs'):
            ground_truth_value = ground_truth_exp.rhs
            print(f"Detected equation, using rhs: {ground_truth_value}")
        else:
            ground_truth_value = ground_truth_exp
            
        # Try to convert SymPy expressions to numerical values
        ground_truth = float(ground_truth_value.evalf())
        student_answer = float(student_answer_exp.evalf())
        
        # Preprocessing: Handle special case where correct answer is 0
        if ground_truth == 0:
            if student_answer == 0:
                return 100
            else:
                    return 0
        
        # Sign consistency check
        if ground_truth * student_answer < 0:
            return 0
        
        # Calculate errors
        absolute_error = abs(student_answer - ground_truth)
        relative_error = absolute_error / abs(ground_truth)

        
        # Judge
        is_extremely_close = (relative_error <= RelTol_100_strict)
        if is_extremely_close:
            return 100
        elif relative_error <= RelTol_90:
            return 90
        elif relative_error <= RelTol_80:
            return 80  
        # None of the standards are met
        else:
            return 0
            
    except Exception as e:
        print(f"  -> numeric_score_calc error: {e}")
        # If numerical conversion fails, fall back to the original scoring method
        return 0

@timeout_decorator.timeout(30, timeout_exception=TimeoutError)
def simplify_with_timeout(expr):
    return simplify(expr)
def time_simplify(expr):
    try:
        result=simplify_with_timeout(expr)
        return result
    except TimeoutError:
        return expr

@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def equal_with_timeout(expr1,expr2):
    return expr1.equals(expr2)
def time_equal(expr1,expr2):
    try:
        result=equal_with_timeout(expr1,expr2)
        return result
    except TimeoutError:
        return False


def sympy_to_tree(expr):
    """
    Convert a SymPy expression into a tree structure.
    This function takes a SymPy expression and recursively converts it into a tree
    representation using `TreeNode` objects. Each node in the tree is labeled based
    on the type of the SymPy expression (e.g., number, symbol, operator, or function),
    and its children represent the arguments of the expression.
    Args:
        expr (sympy.Basic): The SymPy expression to be converted.
    Returns:
        TreeNode: The root node of the tree representation of the SymPy expression.
    Raises:
        ValueError: If the SymPy expression contains an unsupported type.
    Supported Types:
        - Numbers: Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity
        - Symbols: Symbol
        - Binary Operators: Add, Mul, Pow
        - Functions: Any subclass of `sympy.Function`
    Example:
        >>> from sympy import symbols, sin, pi
        >>> x, y = symbols('x y')
        >>> expr = x + y * sin(pi)
        >>> tree = sympy_to_tree(expr)
        >>> print(tree)
    """
  

    """Convert the sympy expression to a tree"""
    if isinstance(expr, MatrixBase):
        children = []
        for i in range(expr.rows):
            for j in range(expr.cols):
                children.append(sympy_to_tree(expr[i, j]))
        return TreeNode(label=f"matrix_{expr.rows}x{expr.cols}", children=children)

    elif isinstance(expr, (Integer, Pi, Exp1, ImaginaryUnit, Float, Rational, Infinity, NegativeInfinity, NaN, ComplexInfinity)):
        return TreeNode(label="number_" + str(expr), children=[])
    elif isinstance(expr, Symbol):
        return TreeNode(label="symbol_" + str(expr), children=[])
    elif isinstance(expr, (Add, Mul, Pow)):
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="operator_" + op_name, children=children)
    elif isinstance(expr, Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="function_" + func_name, children=children)
    elif isinstance(expr, Relational):
        op_name = type(expr).__name__
        children = [sympy_to_tree(expr.lhs), sympy_to_tree(expr.rhs)]
        return TreeNode(label="relation_" + op_name, children=children)
    elif isinstance(expr, Derivative):
        children = [sympy_to_tree(expr.expr)] + [sympy_to_tree(v) for v in expr.variables]
        return TreeNode(label="function_Derivative", children=children)
    elif isinstance(expr, And):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="logic_And", children=children)
    elif isinstance(expr, Or):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="logic_Or", children=children)
    elif isinstance(expr, Not):
        children = [sympy_to_tree(expr.args[0])]
        return TreeNode(label="logic_Not", children=children)
    else:
        raise ValueError(f"Unsupported SymPy type: {type(expr)} Expression: {expr}")

class TreeNode:
    def __init__(self, label, children=None,node_type='other'):
        self.label = label
        self.children = children if children is not None else []
        self.node_type=node_type
        self.subtree_size=0
    def get_children(self):
        return self.children
    
    def __str__(self):
        return self.label

def print_tree(node, indent=0):
    """Print a tree structure"""
    print('  ' * indent + f'└─ {node.label}')
    for child in node.children:
        print_tree(child, indent + 1)

class LaTeXError(Exception):
    def __init__(self, message="LaTeXError"):
        super().__init__(message)

class SymPyError(Exception):
    def __init__(self, message="SymPyError"):
        super().__init__(message)

class TreeError(Exception):
    def __init__(self, message="TreeError"):
        super().__init__(message)

class DistError(Exception):
    def __init__(self, message="DistanceError"):
        super().__init__(message)

def Equation_standardize(latex):
    """
    Standardize equation by converting it to difference form
    """
    return latex.args[0] - latex.args[1]

def extract_interval(latex):
    """
    Extract interval notation from LaTeX string
    Use regular strings (not raw strings), so all backslashes are escaped with \\
    """
    interval_pattern = re.compile(
        r"^\s*"                                 # Leading whitespace
        r"(?:\\left)?\s*"                     # Optional \left
        r"([\(\[])\s*"                          # Group 1: left bracket
        r"(.*?)\s*,\s*"                         # Group 2: lower bound
        r"(.*?)\s*"                             # Group 3: upper bound
        r"(?:\\right)?\s*"                    # Optional \right
        r"([\)\]])\s*$"                         # Group 4: right bracket
    )
    match = interval_pattern.match(latex)
    if match:
        left_bracket, lower_bound, upper_bound, right_bracket = match.groups()
        return True, left_bracket, lower_bound, upper_bound, right_bracket
    else:
        return False, None, None, None, None
    
def judge_interval(latex):
    """
    Judge if a LaTeX string represents an interval
    """
    latex=latex.replace('$','')
    match, left_bracket, lower_bound, upper_bound, right_bracket = extract_interval(latex)
    if match:
        # Judge whether it's open/closed interval
        is_left_closed = left_bracket == "["
        is_right_closed = right_bracket == "]"
        left_type = "l_c" if is_left_closed else "l_o"
        right_type = "r_c" if is_right_closed else "r_o"
        return True, left_type + lower_bound + "+" + upper_bound + right_type
    else:
        return False, latex

def check_latex_wrap(s):
    s = s.strip()
    pattern = r'''
        ^(
            \(.*\) |                            # Regular parentheses ( )
            \[.*\] |                            # Regular square brackets [ ]
            \\\(.*\\\) |                        # LaTeX inline math: \( \)
            \\\[.*\\\] |                        # LaTeX display math: \[ \]
            \\\\left\(.*\\\\right\) |           # LaTeX \left( \right)
            \\\\left\[.*\\\\right\] |           # LaTeX \left[ \right]
            \$.*\$                              # LaTeX inline math with $...$
        )$
    '''
    return re.match(pattern, s, re.VERBOSE) is not None

def parse_bracketed_string(s):
    # Remove surrounding brackets: supports (), \left( \right)
    s = s.strip()
    s = re.sub(r'^\\left\(|^\(', '', s)
    s = re.sub(r'\\right\)$|\)$', '', s)
    parts = [item.strip() for item in s.split(',')]
    return parts

def strip_dollar_signs(s):
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s

def extract_numeric_part(latex_str: str) -> str:
    """
    Numeric extractor
    Intelligently extracts and returns a clean string containing only numbers and basic operators
    from a complex LaTeX string that may contain units, variables, equations.
    """
    if not isinstance(latex_str, str) or not latex_str:
        return ""
                
    s = latex_str.strip()

    # Strip outer LaTeX math environment delimiters
    if s.startswith('$') and s.endswith('$'):
        s = s.strip('$').strip()
    if s.startswith('\\(') and s.endswith('\\)'):
        s = s[2:-2].strip()
    if s.startswith('\\[') and s.endswith('\\]'):
        s = s[2:-2].strip()
    """                 
    If there's an equation or approximately equal sign, take only the right side
    Use non-greedy matching .*? to ensure it doesn't accidentally match too much
    Support various forms like a = b, a \\approx b, etc.
    """
    equal_sign_pattern = r'.*(?:=|\\approx|\\sim|\\simeq|\\propto)\s*(.*)'
    match = re.search(equal_sign_pattern, s)
    if match:
        s = match.group(1).strip()

    # Remove LaTeX whitespace commands so signs adjacent to numbers are preserved
    try:
        s = _remove_latex_whitespace_commands(s)
    except Exception:
        pass
    # Normalize percent: turn "number\%" or "number%" into "number/100"
    s = re.sub(r"(\d(?:[\d\.]*)?)\s*\\%", r"(\1/100)", s)
    s = re.sub(r"(\d(?:[\d\.]*)?)\s*%", r"(\1/100)", s)
    # Remove stray backslashes directly before a sign or digit (e.g., \, -\,2.14 -> -2.14)
    s = re.sub(r'\\(?=[\d\+\-])', '', s)

    """  
    Actively match and extract scientific notation or regular numbers
    This regex can match various forms like -1.28, 1.28e-5, -1.28 \\times 10^{-5}, -1.28 \\\\times 10^{-5}, etc.
    Also normalize common \\frac forms into a/b for rational parsing.
    """  
    # Normalize \\frac forms to a/b to support rational parsing
    # \\frac{a}{b}
    s = re.sub(r"\\frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}", r"(\1)/(\2)", s)
    # \\frac a b (brace-less) for simple numeric tokens
    s = re.sub(r"\\frac\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))", r"\1/\2", s)
    # \\frac12 (compact) -> 1/2
    s = re.sub(r"\\frac\s*([0-9])\s*([0-9])", r"\1/\2", s)

    # Prefer fraction pattern a/b first to avoid capturing only the numerator
    frac_match = re.search(r"[-+]?\s*(?:\(?\s*(?:\d+\.?\d*|\.\d+)\s*\)?\s*/\s*\(?\s*(?:\d+\.?\d*|\.\d+)\s*\)?)", s)
    if frac_match:
        return frac_match.group(0).strip()

    # Fall back to scientific/regular number
    numeric_pattern = re.compile(
        r"([-+]?\s*(?:\d+\.?\d*|\.\d+)\s*(?:(?:e|E)\s*[-+]?\s*\d+|\\\\?times\s*10\^\{?[-+]?\d+\}?)?)"
    )
                
    match = numeric_pattern.search(s)
                
    if match:
        # If successful match, directly return the core numeric string
        numeric_part = match.group(0)
        # Clean up by replacing both \\times and \\\\times with *
        cleaned_part = numeric_part.replace('\\\\times', '*').replace('\\times', '*')
        return cleaned_part.strip()
    return s

def extract_tuple(latex):
    """
    A tuple/key-value pair parser.
    Core strategy:
    1. If the expression is in the form `(keys) = (values)`, [ignore] the left `(keys) =` part,
       only take the right `(values)` as the parsing target.
    2. If the expression is just a tuple `(values)`, parse it directly.
    3. Always return a dictionary with numeric indices as keys, like {'0': val1, '1': val2, ...}.
    """
    latex = strip_dollar_signs(latex.strip())
    latex = latex.replace(r'\left', '')
    latex = latex.replace(r'\right', '')

    # Check if there's a top-level '(keys) = (values)' structure
    paren_level = 0
    top_level_equal_index = -1
    for i, char in enumerate(latex):
        if char in '({[': paren_level += 1
        elif char in ')}]': paren_level -= 1
        elif char == '=' and paren_level == 0:
            top_level_equal_index = i
            break 
    # If found this structure, we only focus on the right side of the equals sign
    if top_level_equal_index != -1:
        left_part = latex[:top_level_equal_index].strip()
        right_part = latex[top_level_equal_index+1:].strip()
        # Do a sanity check to ensure both sides of the equals sign look like tuples
        if check_latex_wrap(left_part) and check_latex_wrap(right_part):
            # override the entire expression with the right side
            latex = right_part

    # Parse the final tuple string
    if not check_latex_wrap(latex):
        return {}

    # remove brackets and split by commas
    values = parse_bracketed_string(latex)
    
    # If it's an empty tuple "()", values will be an empty list after parsing
    if not values:
        # Here we return empty dict, the logic in EED will handle it correctly
        return {}

    # Convert value list to dictionary with numeric indices as keys
    return {str(i): v for i, v in enumerate(values)}

# Unit processing related functions
ureg = pint.UnitRegistry()

def _remove_latex_whitespace_commands(text: str) -> str:
    """Remove common LaTeX whitespace commands from text (no regex side-effects)."""
    if not text:
        return text
    commands = [
        "\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad", "\\thinspace", "\\enspace", "\\ ",
    ]
    for cmd in commands:
        text = text.replace(cmd, "")
    return text

def _safe_parse_numeric_string(numeric_str: str) -> float:
    """
    Safely parse a numeric string that may be in forms like:
    - 1.23
    - -0.5
    - 1e-3 / 1E+6
    - 1.2*10^3 / 1.2 * 10^{3}
    Never uses eval. Returns float or raises ValueError.
    """
    if not isinstance(numeric_str, str):
        raise ValueError("numeric_str must be a string")
    s = numeric_str.strip()
    # Normalize spacing and variants
    s = s.replace("\\times", "*").replace("\\\\times", "*")
    s = re.sub(r"\s+", "", s)
    # Expand percent to division by 100 if trailing
    s = re.sub(r"^(.*?)(\d(?:[\d\.]*)?)/?100\)?$", r"\1(\2/100)", s) if False else s
    if s.endswith('%'):
        s = s[:-1] + "/100"
    if s.endswith('\\%'):
        s = s[:-2] + "/100"
    # Normalize *10^{n} to *10**n
    s = re.sub(r"\*10\^\{?([+-]?\d+)\}?", r"*10**\1", s)
    # Pattern a*10**b
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\*10\*\*([+-]?\d+)", s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    # Pattern scientific e/E
    m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))[eE]([+-]?\d+)", s)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10 ** exp)
    # Fraction a/b (allow simple parentheses around parts), only when exactly one '/'
    if s.count('/') == 1:
        num_str, den_str = s.split('/', 1)
        # strip one layer of parentheses if present
        num_str = re.sub(r"^\((.*)\)$", r"\1", num_str)
        den_str = re.sub(r"^\((.*)\)$", r"\1", den_str)
        num = _safe_parse_numeric_string(num_str)
        den = _safe_parse_numeric_string(den_str)
        if den == 0:
            raise ValueError("Division by zero in fraction")
        return num / den
    # Plain number
    m = re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", s)
    if m:
        return float(s)
    raise ValueError(f"Unrecognized numeric format: {numeric_str}")

def clean_latex_unit(unit_str):
    r"""
    Clean LaTeX unit string for pint parsing
    Recursively clean LaTeX wrapping like \mathrm{}, \text{}, \operatorname{} from unit strings,
    extract plain text units while preserving braces in exponent parts.
    """
    pattern = re.compile(r"\\(mathrm|text|operatorname)\{([^{}]*(\{[^{}]*\}[^{}]*)*)\}")
    prev_str = None
    while prev_str != unit_str:
        prev_str = unit_str
        unit_str = pattern.sub(r"\2", unit_str)
    if unit_str.startswith("{") and unit_str.endswith("}"):
        unit_str = unit_str[1:-1]
    unit_str = unit_str.strip()
    unit_str = _remove_latex_whitespace_commands(unit_str)
    return unit_str

def parse_latex_quantity_general(latex_str):
    r"""
    Generically parse LaTeX-formatted quantity strings to extract numeric values and units.
    Supports:
    - Numbers (including decimals, negative signs, scientific notation, and LaTeX-style scientific notation)
    - Units wrapped in \mathrm{} or \text{}, or without any wrapper
    - Removal of all LaTeX whitespace commands
    Returns: (float value, unit string)
    """
    numeric_part = extract_numeric_part(latex_str)
    try:
        number = _safe_parse_numeric_string(numeric_part)
    except Exception as e:
        raise ValueError(f"Failed to compute numeric value from: {numeric_part}, error: {e}")

    original_numeric = re.search(r"[-+]?\s*(?:\d+\.?\d*|\.\d+)\s*(?:(?:e|E)\s*[-+]?\s*\d+|\\\\?times\s*10\^\{?[-+]?\d+\}?)?", latex_str)
    if original_numeric:
        unit_part = latex_str[original_numeric.end():].strip()
    else:
        unit_part = ""
    
    unit_part = clean_latex_unit(unit_part)
    return number, unit_part

def convert_and_output_general(latex_qty1, latex_qty2, target_unit=None):
    """
    Parse two generalized LaTeX-formatted quantity strings, convert them to the target unit, and output the result.
    If target_unit is empty, convert to the unit of the first quantity.
    """
    n1, u1 = parse_latex_quantity_general(latex_qty1)
    n2, u2 = parse_latex_quantity_general(latex_qty2)

    q1 = n1 * ureg(u1)
    q2 = n2 * ureg(u2)

    if target_unit is None:
        target_unit = u1

    q1_converted = q1.to(target_unit)
    q2_converted = q2.to(target_unit)

    out1 = f"{q1_converted.magnitude} {target_unit}"
    out2 = f"{q2_converted.magnitude} {target_unit}"

    return out1, out2

def SEED(answer_latex,test_latex,expr_type,debug_mode=False):
    """
    SEED (Scalable Expression Edit Distance) - Enhanced version of EED
    NEW FEATURES in SEED vs EED:
    Multi-type expression support: Expression, Equation, Tuple, Interval, Numeric
    Advanced numeric scoring with relative/absolute error thresholds
    Physical unit conversion and comparison using Pint library
    Intelligent tuple/key-value pair parsing and comparison
    Interval notation support with open/closed bracket distinction
    Improved equation standardization (A=B → A-B)
    Robust error handling
    
    Computes the similarity score and distance metrics between two LaTeX expressions.
    
    This function evaluates the equivalence of two mathematical expressions represented 
    in LaTeX format. It uses symbolic computation and tree-based distance metrics to 
    calculate a similarity score and other related metrics.
    
    Args:
        answer_latex: The latex expression of answer expression
        test_latex: The latex expression of test expression
        t: Expression type (Expression, Equation, Tuple, Interval, Numeric)
        debug_mode: Whether it raise errors or just skip it
    
    Returns:
        tuple: A tuple containing the following elements:
            - score (float): The similarity score between the two expressions (0 to 100).
            - relative_distance (float): The normalized distance between the two expressions.
            - answer_tree_size (int): The size of the expression tree for the answer.
            - distance (float): The raw distance between the two expression trees.
    
    Notes:
        - If either input contains unsupported LaTeX constructs (e.g., integrals or sums), 
          the function returns default values indicating failure.
        - If the test expression is significantly longer than the answer expression, 
          the function assumes they are not equivalent.
        - The function uses symbolic simplification and tree-based distance metrics to 
          evaluate equivalence.
        - In case of errors during processing, the function returns default values unless 
          `debug_mode` is enabled, in which case it raises specific exceptions.
    
    Exceptions:
        - LaTeXError: Raised when LaTeX conversion to symbolic expressions fails (if `debug_mode` is True).
        - SymPyError: Raised when symbolic simplification or tree construction fails (if `debug_mode` is True).
        - DistError: Raised when distance calculation fails (if `debug_mode` is True).
    """

    if not test_latex:
        return 0,-1,-1,-1
    if '\\int' in test_latex or '\\int' in answer_latex:
        return 0,-1,-1,-1
    if '\\sum' in test_latex or '\\sum' in answer_latex:
        return 0,-1,-1,1
    if answer_latex==test_latex:
        return 100,0.0,-1,0
    # if len(test_latex)>3*len(answer_latex):
    #     return 0,-1,-1,-1
    
    try:
        if expr_type == 'Tuple':
            answer_dict = extract_tuple(answer_latex)
            test_dict = extract_tuple(test_latex)

            if not answer_dict or not test_dict:
                return 0, -1, -1, -1

            try:
                norm_answer_dict = {master_convert(k, 'Expression'): v for k, v in answer_dict.items()}
                norm_test_dict = {master_convert(k, 'Expression'): v for k, v in test_dict.items()}
            except Exception as e:
                if debug_mode: print(f"Error normalizing tuple keys: {e}")
                return 0, -1, -1, -1

            if set(norm_answer_dict.keys()) != set(norm_test_dict.keys()):
                return 0, -1, -1, -1

            scores, rel_distances, tree_sizes, distance_numbers = 0, 0, 0, 0
            size = len(norm_answer_dict)
            if size == 0:
                return 100, 0.0, 0, 0

            for sympy_key, answer_v_latex in norm_answer_dict.items():
                test_v_latex = norm_test_dict[sympy_key]
                
                # Recursively call to compare SEED values
                score, rel_distance, tree_size, distance_number = SEED(answer_v_latex, test_v_latex, 'Expression')
                scores += score

                if rel_distance != -1: rel_distances += rel_distance
                if tree_size != -1: tree_sizes += tree_size
                if distance_number != -1: distance_numbers += distance_number

            return scores / size, rel_distances / size, tree_sizes / size, distance_numbers / size
        
        elif expr_type=='Interval':
            is_interval, answer_latex= judge_interval(answer_latex)
            is_interval, test_latex= judge_interval(test_latex)
            # if is_interval:t='Interval'
        elif expr_type=='Numeric':
            # Numeric path: directly compute numeric values first using SymPy on RHS, then try units, then fallback
            def _rhs_or_self(s: str) -> str:
                ss = s.strip()
                if ss.startswith('$') and ss.endswith('$'):
                    ss = ss.strip('$').strip()
                if ss.startswith('\\(') and ss.endswith('\\)'):
                    ss = ss[2:-2].strip()
                if ss.startswith('\\[') and ss.endswith('\\]'):
                    ss = ss[2:-2].strip()
                m = re.search(r'.*(?:=|\\approx|\\sim|\\simeq|\\propto)\s*(.*)', ss)
                if m:
                    return m.group(1).strip()
                return ss

            def _normalize_numeric_rhs(s: str) -> str:
                # Replace common LaTeX multiply operators and remove whitespace commands
                s = re.sub(r'\\+times', '*', s)
                s = re.sub(r'\\+cdot',  '*', s)
                s = _remove_latex_whitespace_commands(s)
                return s

            try:
                ans_rhs = _normalize_numeric_rhs(_rhs_or_self(answer_latex))
                tst_rhs = _normalize_numeric_rhs(_rhs_or_self(test_latex))
                ans_exp_try = master_convert(ans_rhs, 'Expression')
                test_exp_try = master_convert(tst_rhs, 'Expression')
                if ans_exp_try is not None and test_exp_try is not None:
                    try:
                        if getattr(ans_exp_try, 'free_symbols', set()) or getattr(test_exp_try, 'free_symbols', set()):
                            pass  # fall through to unit-aware parsing
                        else:
                            score = numeric_score_calc(test_exp_try, ans_exp_try)
                            return score, -1, -1, -1
                    except Exception:
                        pass
            except Exception:
                pass

            def _try_parse_quantity(s):
                try:
                    return parse_latex_quantity_general(s)
                except Exception:
                    return None, None
            a_val, a_unit = _try_parse_quantity(answer_latex)
            t_val, t_unit = _try_parse_quantity(test_latex)

            if a_val is not None and t_val is not None and a_unit and t_unit:
                try:
                    qa = a_val * ureg(a_unit)
                    qt = t_val * ureg(t_unit)
                    qt_conv = qt.to(qa.units)
                    score = numeric_score_calc(Float(qt_conv.magnitude), Float(qa.magnitude))
                    return score, -1, -1, -1
                except Exception:
                    pass

            try:
                if a_val is None:
                    a_val = _safe_parse_numeric_string(extract_numeric_part(answer_latex))
                if t_val is None:
                    t_val = _safe_parse_numeric_string(extract_numeric_part(test_latex))
                print(a_val)
                print(t_val)
                score = numeric_score_calc(Float(t_val), Float(a_val))
                return score, -1, -1, -1
            except Exception:
                return 0, -1, -1, -1

        answer_exp = master_convert(answer_latex, expr_type)
        test_exp = master_convert(test_latex, expr_type)
        if expr_type =='Equation':
            answer_exp = Equation_standardize(answer_exp)
            test_exp = Equation_standardize(test_exp)

    except Exception as e:
        if debug_mode:
            raise LaTeXError(f"Fail to convert latex.\n GT:{answer_latex}\n GEN:{test_latex}")
        return 0,-1,-1,-1

    try:
        if answer_exp is None or test_exp is None:
            return 0,-1,-1,-1
        answer_exp,rep1=posify(answer_exp)
        answer_exp=time_simplify(answer_exp)
        
        test_exp,rep2=posify(test_exp)
        test_exp=time_simplify(test_exp)

        answer_exp=answer_exp.subs(rep1)
        test_exp=test_exp.subs(rep2)

        # if False:
        @timeout_decorator.timeout(10, timeout_exception=TimeoutError)
        def subtract_and_simplify_with_timeout(a, b):
            if isinstance(a, Expr) and isinstance(b, Expr):
                return simplify(expand(a - b))
            elif isinstance(a, Matrix) and isinstance(b, Matrix):
                if a.shape == b.shape:
                    return simplify(expand(a - b))
                else:
                    return 1  # Matrix dimensions do not match
            else:
                return 1
        
        def safe_subtract_and_simplify(a, b):
            try:
                return subtract_and_simplify_with_timeout(a, b)
            except TimeoutError:
                print("  -> subtract_and_simplify timeout, returning 1")
                return 1  # Treat as unequal if a timeout occurs
            except Exception as e:
                print(f"  -> subtract_and_simplify error: {e}")
                return 1
        zero_exp=safe_subtract_and_simplify(answer_exp,test_exp)
        # zero_exp=time_simplify(expand(answer_exp-test_exp))       

        if expr_type == "Equation":
            if answer_exp == test_exp or zero_exp == 0 or answer_exp + test_exp == 0:
                return 100, 0., 0, 0

        if answer_exp == test_exp or zero_exp == 0:
            return 100, 0., 0, 0

        if time_equal(answer_exp, test_exp):
            return 100, 0., 0, 0

    except Exception as e:
        if debug_mode:
            raise SymPyError(f"Failed to simplify the sympy expression. Expressions: answer_exp={answer_exp}, test_exp={test_exp}")
        return 0,-1,-1,-1

    try:
        tree_answer=sympy_to_tree(answer_exp)
        tree_test=sympy_to_tree(test_exp)

    except Exception as e:
        if debug_mode:
            raise SymPyError(f"Failed to build the sympy expression tree.\n GT:{answer_exp}\n GEN:{test_exp}")
        return 0,-1,-1,-1

    distance=ext_distance(
                tree_test,
                tree_answer,
                get_children=lambda x:x.get_children(),
                single_insert_cost=insert_func,
                insert_cost=insert_tree_func,
                single_remove_cost=remove_func, 
                remove_cost=remove_tree_func, 
                update_cost=update_func)    

    tree_size=calc_tree_size(tree_answer)
    distance_number=distance

    rel_distance=distance/tree_size
    
    # Non-numeric types use tree-based scoring
    score = score_calc(distance_number, tree_size)
    return score,rel_distance,tree_size,distance_number
    
if __name__ == "__main__":
    def run_case(idx: int, gt: str, pred: str, expr_type: str, note: str = ""):
        print("\n" + "=" * 80)
        title = f"Case #{idx}  [{expr_type}]"
        if note:
            title += f"  — {note}"
        print(title)
        print("-" * 80)
        try:
            score, rel_distance, tree_size, dist = SEED(gt, pred, expr_type)
        except Exception as e:
            print(f"[ERROR] Exception while scoring: {e}")
            return

        print(f"GT LaTeX:      {gt}")
        print(f"Predicted:     {pred}")
        print(f"Score:         {score}")
        print(f"Rel Distance:  {rel_distance}")
        print(f"Tree Size:     {tree_size}")
        print(f"Raw Distance:  {dist}")

    tests = [
        # ----------------------- Expression -----------------------
        {
            "expr_type": "Expression",
            "gt":  r"2 m g + 4\frac{m v_0^2}{l}",
            "pred": r"2 m g + 4\frac{m v_0^2}{l}",
            "note": "Exact match → expect 100"
        },
        {
            "expr_type": "Expression",
            "gt":  r"2 m g + 4\frac{m v_0^2}{l}",
            "pred": r"2 m g + 2\frac{m v_0^2}{l}",
            "note": "Coefficient differs → partial score"
        },

        # ----------------------- Equation -------------------------
        {
            "expr_type": "Equation",
            "gt":  r"x^2 + 2x + 1 = 0",
            "pred": r"x^2 + 2x + 1 + 0 = 0",
            "note": "Equivalent equation (add 0) → expect 100"
        },
        {
            "expr_type": "Equation",
            "gt":  r"x + y = 0",
            "pred": r"x + y + 0 = 0",
            "note": "Trivially equivalent → expect 100"
        },

        # ----------------------- Tuple ----------------------------
        {
            "expr_type": "Tuple",
            "gt":  r"(x, y, z)",
            "pred": r"\left(x, y, z \right)",
            "note": "Same tuple with \\left/\\right → expect 100"
        },
        {
            "expr_type": "Tuple",
            "gt":  r"(x, y, z)",
            "pred": r"(x, z, y)",
            "note": "Permutation in positions → partial score"
        },

        # ----------------------- Interval -------------------------
        {
            "expr_type": "Interval",
            "gt":  r"[0, 1]",
            "pred": r"\left[0, 1 \right]",
            "note": "Closed interval same form → expect 100"
        },
        {
            "expr_type": "Interval",
            "gt":  r"(a, b]",
            "pred": r"[a, b]",
            "note": "Open/closed boundary differs → likely < 100"
        },

        # ----------------------- Numeric (with units) -------------
        {
            "expr_type": "Numeric",
            "gt":  r"4.2 \times 10^5 \mathrm{m^{2}}",
            "pred": r"0.42 \mathrm{km^{2}}",
            "note": "Unit conversion m^2 ↔ km^2 → expect 100"
        },
        {
            "expr_type": "Numeric",
            "gt":  r"1000 \mathrm{m}",
            "pred": r"1 \mathrm{km}",
            "note": "Unit conversion m ↔ km → expect 100"
        },
        {
            "expr_type": "Numeric",
            "gt":  r"9.81 \mathrm{m/s^{2}}",
            "pred": r"981 \mathrm{cm/s^{2}}",
            "note": "Unit conversion m/s^2 ↔ cm/s^2 → expect 100"
        },
        {
            "expr_type": "Numeric",
            "gt":  r"1.000 \mathrm{m}",
            "pred": r"0.990 \mathrm{m}",
            "note": "≈1% relative error → expect 80"
        },
        {
            "expr_type": "Numeric",
            "gt":  r"3.14",
            "pred": r"3.1400",
            "note": "No units, numerically equal → expect 100"
        },
        {
            "expr_type": "Numeric",
            "gt":  r"5",
            "pred": r"-5",
            "note": "Sign mismatch → expect 0"
        },
    ]

    for i, case in enumerate(tests, 1):
        run_case(i, case["gt"], case["pred"], case["expr_type"], case.get("note", ""))
