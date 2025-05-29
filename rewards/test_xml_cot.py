import os
import re
import json
import string
import random
import unittest
import asyncio as aio
from xml_cot import (
    xml_cot_parse_solution_fn,
    tree_depth, tree_width,
    XMLCoTComputeScore,
    xml_cot_compute_score_valid,
)


def load_xml_cot():
    batch_solution_str, batch_ground_truth = [], []
    with open("/cpfs01/shared/llm_ddd/tongjian/sft/self_improvement/xml_cot_if_enhance_0529.jsonl", "rt") as f:
        for line in f:
            example = json.loads(line)

            batch_solution_str.append(
                example["self_improvement"]["responses"][0]["response"]
            )
            batch_ground_truth.append(
                {
                    "ground_truth": example["self_improvement"]["reference_meta"]["final_answer"],
                    "prompt": example["self_improvement"]["prompt"]
                }
            )
            if len(batch_ground_truth) == 128:
                break

    return batch_solution_str, batch_ground_truth


CASE = """
 ```xml
 <think>
     <approach>
         <subgoal sequence="1">
             <plan>
                 <step sequence="1">Understand the problem: We need to find the two distinct values of \( k \) that satisfy the given conditions, and then compute \( |x^2 - y^2| \).</step>
             </plan>
             <verification method="Problem Understanding">
                 <subgoal sequence="1.1">
                     <plan>
                         <step sequence="1">Verify the triangle side lengths: \( a = 3 \), \( b = 5 \), and \( c = k \).</step>
                         <step sequence="2">Verify the area: \( \text{Area} = 6 \) square units.</step>
                     </plan>
                     <verification method="Boundary Check">
                         Confirmed that the problem is well-defined.
                     </verification>
                 </subgoal>
             </verification>
         </subgoal>
         <subgoal sequence="2">
             <plan>
                 <step sequence="1">Use Heron's formula to find the area of the triangle. Heron's formula states that the area \( A \) of a triangle with sides \( a \), \( b \), and \( c \) is given by:
                 \[
                 A = \sqrt{s(s-a)(s-b)(s-c)}
                 \]
                 where \( s \) is the semi-perimeter:
                 \[
                 s = \frac{a + b + c}{2} = \frac{3 + 5 + k}{2} = \frac{8 + k}{2}
                 \]
                 Given \( A = 6 \), we have:
                 \[
                 6 = \sqrt{\left(\frac{8 + k}{2}\right)\left(\frac{8 + k}{2} - 3\right)\left(\frac{8 + k}{2} - 5\right)\left(\frac{8 + k}{2} - k\right)}
                 \]
                 Simplify the expression inside the square root:
                 \[
                 6 = \sqrt{\left(\frac{8 + k}{2}\right)\left(\frac{8 + k - 6}{2}\right)\left(\frac{8 + k - 10}{2}\right)\left(\frac{8 + k - 2k}{2}\right)}
                 \]
                 \[
                 6 = \sqrt{\left(\frac{8 + k}{2}\right)\left(\frac{2 + k}{2}\right)\left(\frac{-2 + k}{2}\right)\left(\frac{8 - k}{2}\right)}
                 \]
                 Simplify further:
                 \[
                 6 = \sqrt{\frac{(8 + k)(2 + k)(-2 + k)(8 - k)}{16}}
                 \]
                 \[
                 6 = \frac{\sqrt{(8 + k)(2 + k)(-2 + k)(8 - k)}}{4}
                 \]
                 Multiply both sides by 4:
                 \[
                 24 = \sqrt{(8 + k)(2 + k)(-2 + k)(8 - k)}
                 \]
                 Square both sides:
                 \[
                 576 = (8 + k)(2 + k)(-2 + k)(8 - k)
                 \]
             </plan>
             <verification method="Heron's Formula Application">
                 <subgoal sequence="2.1">
                     <plan>
                         <step sequence="1">Verify the application of Heron's formula:</step>
                         <step sequence="2">Simplify the semi-perimeter expression:</step>
                         <step sequence="3">Verify the simplified area expression:</step>
                     </plan>
                     <verification method="Algebraic Simplification">
                         Confirmed the application of Heron's formula and the simplification steps.
                     </verification>
                 </subgoal>
             </verification>
         </subgoal>
         <subgoal sequence="3">
             <plan>
                 <step sequence="1">Expand the left-hand side of the equation:
                 \[
                 576 = (8 + k)(2 + k)(-2 + k)(8 - k)
                 \]
                 Let \( x = 8 + k \) and \( y = 8 - k \). Then:
                 \[
                 576 = (x)(y)(y - 4)(x - 4)
                 \]
                 Notice that \( x + y = 16 \) and \( x - y = 2k \). We can express \( k \) in terms of \( x \) and \( y \):
                 \[
                 k = \frac{x - y}{2}
                 \]
                 Substitute \( k = \frac{x - y}{2} \) into the equation:
                 \[
                 576 = (x)(y)(y - 4)(x - 4)
                 \]
                 Since \( x + y = 16 \), we can express \( y \) in terms of \( x \):
                 \[
                 y = 16 - x
                 \]
                 Substitute \( y = 16 - x \) into the equation:
                 \[
                 576 = x(16 - x)(16 - x - 4)(x - 4)
                 \]
                 Simplify:
                 \[
                 576 = x(16 - x)(12 - x)(x - 4)
                 \]
                 Let \( z = x - 8 \). Then:
                 \[
                 x = z + 8
                 \]
                 Substitute \( x = z + 8 \) into the equation:
                 \[
                 576 = (z + 8)(8 - z)(4 - z)(z - 4)
                 \]
                 Simplify:
                 \[
                 576 = (z + 8)(8 - z)(z - 4)(4 - z)
                 \]
                 Let \( w = z - 4 \). Then:
                 \[
                 z = w + 4
                 \]
                 Substitute \( z = w + 4 \) into the equation:
                 \[
                 576 = (w + 12)(-w)(w)(-w)
                 \]
                 Simplify:
                 \[
                 576 = (w + 12)(w^3)
                 \]
                 This is a cubic equation in \( w \). To solve for \( w \), we need to find the roots of the equation:
                 \[
                 w^4 + 12w^3 - 576 = 0
                 \]
                 We can use numerical methods or factorization to find the roots. The roots of the equation are \( w = -6 \) and \( w = 6 \). Therefore:
                 \[
                 z = -6 + 4 = -2 \quad \text{or} \quad z = 6 + 4 = 10
                 \]
                 Since \( z = x - 8 \), we have:
                 \[
                 x = -2 + 8 = 6 \quad \text{or} \quad x = 10 + 8 = 18
                 \]
                 Therefore:
                 \[
                 k = \frac{6 - 16}{2} = -5 \quad \text{or} \quad k = \frac{18 - 16}{2} = 1
                 \]
                 The two distinct values of \( k \) are \( x = 1 \) and \( y = -5 \).
             </plan>
             <verification method="Algebraic Simplification">
                 <subgoal sequence="3.1">
                     <plan>
                         <step sequence="1">Verify the expansion of the left-hand side:</step>
                         <step sequence="2">Substitute \( k = \frac{x - y}{2} \) into the equation:</step>
                         <step sequence="3">Substitute \( y = 16 - x \) into the equation:</step>
                         <step sequence="4">Simplify the equation:</step>
                         <step sequence="5">Solve the cubic equation:</step>
                     </plan>
                     <verification method="Root Finding">
                         Confirmed the algebraic simplification and the roots of the cubic equation.
                     </verification>
                 </subgoal>
             </verification>
         </subgoal>
         <subgoal sequence="4">
             <plan>
                 <step sequence="1">Compute \( |x^2 - y^2| \):
                 \[
                 x = 1 \quad \text{and} \quad y = -5
                 \]
                 \[
                 x^2 = 1^2 = 1 \quad \text{and} \quad y^2 = (-5)^2 = 25
                 \]
                 \[
                 |x^2 - y^2| = |1 - 25| = 24
                 \]
             </plan>
             <verification method="Final Calculation">
                 <subgoal sequence="4.1">
                     <plan>
                         <step sequence="1">Calculate \( x^2 \) and \( y^2 \):</step>
                         <step sequence="2">Compute the absolute difference:</step>
                     </plan>
                     <verification method="Numerical Verification">
                         Confirmed the final calculation.
                     </verification>
                 </subgoal>
             </verification>
         </subgoal>
         <subgoal sequence="5">
             <plan>
                 <step sequence="1">Verify the solution:
                 - The triangle with side lengths \( 3 \), \( 5 \), and \( k \) has an area of \( 6 \).
                 - The values of \( k \) that satisfy this condition are \( x = 1 \) and \( y = -5 \).
                 - The value of \( |x^2 - y^2| \) is \( 24 \).
             </plan>
             <verification method="Solution Verification">
                 <subgoal sequence="5.1">
                     <plan>
                         <step sequence="1">Check the triangle inequality:</step>
                         <step sequence="2">Verify the area condition:</step>
                         <step sequence="3">Confirm the final result:</step>
                     </plan>
                     <verification method="Boundary Check">
                         Confirmed the triangle inequality and the area condition.
                     </verification>
                 </subgoal>
             </verification>
         </subgoal>
     </approach>
 </think>
 <conclusion>
     Answer: 24
 </conclusion>
```
"""


class TestXMLCoT(unittest.TestCase):
    def test_parse(self):
        print(xml_cot_parse_solution_fn(CASE))

    def test_compute_score(self):
        task = XMLCoTComputeScore(split="valid")
        batch_solution_str, batch_ground_truth = load_xml_cot()
        print(task.compute_score(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )), len(batch_solution_str)

    def test_thought_reward(self):
        batch_solution_str, batch_ground_truth = load_xml_cot()
        # task = XMLCoTComputeScore(split="valid")
        # results = task.thought_reward(
        #     [None] *
        #     len(batch_solution_str), batch_solution_str, batch_ground_truth
        # )
        # print(results)
        results = xml_cot_compute_score_valid(
            [None] *
            len(batch_solution_str), batch_solution_str, batch_ground_truth
        )
        print(len(results), len(batch_solution_str))


if __name__ == '__main__':
    unittest.main()
