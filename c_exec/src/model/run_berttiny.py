# Load model directly
from transformers import AutoTokenizer, BertForQuestionAnswering
import torch
from torch.profiler.profiler import *
from torch.profiler import *
import os

do_profile = True

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
    print(model)

    question, text = ("What's the biggest problem with using these new neural network calculations?",
                      "Looking at these new neural network calculations, what’s remarkable is that they’re essentially "
                      "a return to a failed project of nearly 40 years ago. "
                      "In 1985 the exciting new idea was that maybe compactifying a 10d superstring on a Calabi-Yau "
                      "would give the Standard Model. "
                      "It quickly became clear that this wasn’t going to work. "
                      "A minor problem was that there were quite a few classes of Calabi-Yaus, but the really big "
                      "problem was that the Calabi-Yaus in each class were parametrized by a large dimensional moduli "
                      "space. One needed some method of “moduli stabilization” that would pick out specific moduli "
                      "parameters. Without that, the moduli parameters became massless fields, introducing a huge host"
                      " of unobserved new long-range interactions. The state of the art 20 years later is that endless "
                      "arguments rage over whether Rube Goldberg-like constructions such as KKLT can consistently "
                      "stabilize moduli (if they do, you get the “landscape” and can’t calculate anything anyway, since "
                      "these constructions give exponentially large numbers of possibilities). ")

    # multiply the question and text above to make a large batch of 40
    question = [question] * 40
    text = [text] * 40

    inputs = tokenizer(question, text, return_tensors="pt")

    print("data type is " + str(inputs["input_ids"].dtype))
    print("model parameter data type is " + str(next(model.parameters()).dtype))

    # get highest log number in the log directory
    log_num = 0
    prev_logs = [int(a.split("_")[-1]) for a in list(os.listdir("log"))]
    if len(prev_logs) > 0:
        log_num = max(prev_logs) + 1

    if do_profile:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True,
                     with_flops=True, with_modules=True,
                     on_trace_ready=tensorboard_trace_handler(f'./log/bert-tiny_{log_num}'),
                     use_cuda=False) as prof:  #  if you want a tensorboard log
            with record_function("model_inference"):
                with torch.no_grad():
                    outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        print(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))

