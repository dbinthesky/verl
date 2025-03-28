import os
import json
import random
import unittest
import numpy as np
from rm_w_criteria import (
    compute_rm_score,
    postprocess_solution,
    simple_tokenize
)


def load_xml_cot_data(num):
    TEST_CASE = "/cpfs01/shared/llm_ddd/tongjian/ddm/thought_xml/verify_enhance/xml_verify_enhance_v2.jsonl"

    batch_solution_str, batch_ground_truth = [], []
    correct_indices, wrong_indices = [], []
    with open(TEST_CASE, "rt") as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            try:
                prompt = example["self_improvement"]["prompt"]
                corrects = example["self_improvement"]["responses"]
                wrongs = example["self_improvement"]["wrong_responses"]
                if len(corrects) > 0 and len(wrongs) > 0:
                    correct = random.choice(corrects)
                    wrong = random.choice(wrongs)

                    batch_solution_str.append(correct["conclusion"])
                    batch_ground_truth.append({
                        "ground_truth": f'{prompt}\n\n\n\nJudge only by determining whether the final answer is correct.\n** Final Answer: {example["self_improvement"]["reference_meta"]["final_answer"]}',
                        "extra_info": {
                            "question_type": "object"
                        }
                    })
                    correct_indices.append(len(batch_ground_truth)-1)

                    batch_solution_str.append(wrong["conclusion"])
                    batch_ground_truth.append({
                        "ground_truth": f'{prompt}\n\n\n\nJudge only by determining whether the final answer is correct.\n** Final Answer: {example["self_improvement"]["reference_meta"]["final_answer"]}',
                        "extra_info": {
                            "question_type": "object"
                        }
                    })
                    wrong_indices.append(len(batch_ground_truth)-1)

            except Exception as err:
                continue
            if i > num:
                break
        return batch_solution_str, batch_ground_truth, correct_indices, wrong_indices


class TestRMReward(unittest.TestCase):
    def test_grid_search_rm_threshold(self):
        num = 10000
        threshold = 0.0
        batch_solution_str, batch_ground_truth, correct_indices, wrong_indices = load_xml_cot_data(
            num)

        precision, recall = [], []
        for i, score in enumerate(compute_rm_score(batch_solution_str, batch_ground_truth, postprocess_solution)):
            if score >= threshold:
                if i in correct_indices:
                    precision.append(1.)
                else:
                    precision.append(0.)
            else:
                if i in wrong_indices:
                    recall.append(1.)
                else:
                    recall.append(0.)
        print(f'Precision={np.mean(precision)*100:.2f}% ({len(precision)})')
        print(f'Recall={np.mean(recall)*100:.2f}% ({len(recall)})')

    def test_solution_len_analysis(self):
        num = 10000
        batch_solution_str, batch_ground_truth, correct_indices, wrong_indices = load_xml_cot_data(
            num)

        size = []
        for i, (solution_str, ground_truth) in enumerate(zip(batch_solution_str, batch_ground_truth)):
            size.append(len(simple_tokenize(solution_str)))
        size = sorted(size)
        for _ in (0.85, 0.9, 0.95):
            print(f'{_}% Length={size[int(len(size)*_)]}')

    # grid_search_rm_threshold()
    # solution_len_analysis()
    # ConclusionTooLongPenalty()

        # # client = get_client(model_type="OPENAI")
        # # print(robust_call_api(client, prompt='你是谁？', model="gpt-4o"))

        # client = get_client(model_type="SELF_DEPLOY", ip=os.getenv(
        #     "DEFAULT_DEPLOY_BASE_URL").split("/")[-2].split(":")[0])
        # while True:
        #     prompt = random.choice([
        #         "What is the major product of the reaction when 3-methyl-1-butene reacts with HBr in the presence of peroxides?",
        #         "Ethyl acetate is treated with benzophenone (a photochemical reagent) under UV light, followed by a reducing agent. Determine the final product of this reaction sequence. Show the change in the oxidation state of the carbonyl carbon in ethyl acetate at each step of the reaction."
        #         "What is the contact angle on the rough surface if the contact angle on the smooth surface is 110°, the roughness \\( r \\) is 1.2, and the rough surface is 80% hydrophobic?"
        #         "What is the transmissivity of a laser cavity with two highly reflective mirrors, where the optical thickness is 4.6 times the wavelength of the light?"
        #         "Ethyl acetate is treated with benzophenone (a photochemical reagent) under UV light, followed by a reducing agent. Determine the final product of this reaction sequence. Show the change in the oxidation state of the carbonyl carbon in ethyl acetate at each step of the reaction."
        #         "Cas9 nuclease and the restriction enzyme EcoRI are produced by the bacterial system. Which of the following statements are true regarding both enzymes?\n I. They are both endonucleases that perform cleavage activity at the specific nucleotide sequence\nII. They both create double-strand breaks in DNA. \nIII. Both Cas9 and EcoRI cut double-stranded DNA forming overhangs in the DNA. \nIV. They are both part of bacterial defense mechanisms against foreign DNA\n\nA. I, II, III\nB. I, III, IV\nC. I, II, III, IV\nD. I, II, IV"
        #     ])
        #     print(prompt)
        #     print('-'*80)
        #     print(robust_call_api(client, prompt=prompt,
        #           model=os.getenv("DEFAULT_DEPLOY_MODEL_NAME")).response)
        #     print("="*80)
        #     # break


if __name__ == '__main__':
    unittest.main()
