# Overview

This is an implementation of EleutherAI
[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B). The model consists of 28 layers with a model dimension of 4096,
and a feedforward dimension of 16384. The model dimension is split into 16 heads, each with a dimension of 256.
Rotary Position Embedding (RoPE) is applied to 64 dimensions of each head. The model is trained with a tokenization
vocabulary of 50257, using the same set of BPEs as GPT-2/GPT-3.

# Deploying to Baseten

To deploy this Truss on Baseten, first install the Baseten client:

```
$ pip install baseten
```

Then, in a Python shell, you can do the following to have an instance of CLIP deployed
on Baseten:

```python
import baseten
import truss

gpt_j_handle = truss.load(".")
baseten.deploy(gpt_j_handle, model_name="GPT-J")
```

# Usage

## Inputs
The input should be a list of dictionaries and must contain the following key:
* `prompt` - the prompt for text generation

Additionally; the following optional parameters are supported as pass thru to the `generate` method. For more details
 look towards the [official documentation](https://huggingface.co/docs/transformers/main/en/main_classes/
text_generation#transformers.generation_utils.GenerationMixin.generate)

* `max_length` - int - limited to  512
* `min_length` - int - limited to 64
* `do_sample` - bool
* `early_stopping` - bool
* `num_beams` - int
* `temperature`  - float
* `top_k` - int
* `top_p` - float
* `repetition_penalty` - float
* `length_penalty` - float
* `encoder_no_repeat_ngram_size` - int
* `num_return_sequences` - int
* `max_time` - float
* `num_beam_groups` - int
* `diversity_penalty` - float
* `remove_invalid_values` - bool

Here's an example input:

```json
[
    {
        "prompt": "If I was a billionaire, I would",
        "max_length": 50
    }
]
```

## Outputs
The result will be a dictionary containing:
* `status` - either `success` or `failed`
* `data` - the output text
* `message` - will contain details in the case of errors

```json
{"status": "success", "data": "If I was a billionaire, I would buy a plane.", "message": null}
```

## Example

You can invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
 curl -X POST https://app.staging.baseten.co/models/{MODEL_VERSION_ID}/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "Answer the question: What is 1+1"}'
{"model_id": "v0VV40K", "model_version_id": "7qrmz03", "model_output": {"status": "success", "data": "Answer the question: What is 1+1?\n\nThe answer is 2.\n\nThe", "message": null}}
```