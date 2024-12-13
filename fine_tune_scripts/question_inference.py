# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BartForConditionalGeneration, AutoModelForSeq2SeqLM

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("likhithasapu/codemixed-airavata")
model = AutoModelForCausalLM.from_pretrained("likhithasapu/codemixed-airavata", device_map="auto")


texts = [
            "A dell."
        ]

input_text = [f"""<|user|> 
              Generate a Hindi-English pun question for the given sentence: <s> What's up दूध? </s>
              American milk ने Indian milk से क्या कहा?
              
              Generate a Hindi-English pun question for the given sentence: <s> They mine their own business </s>
              Coal miners हमेशा खुश क्यों रहते हैं?
              
              Generate a Hindi-English pun question for the given sentence: <s> Love means nothing to them </s>
              किसी tennis player को date करना एक बुरा idea क्यों है?
              
              Generate a Hindi-English pun question for the given sentence: <s> He was toad away </s>
              क्या आपने एक frog के बारे में सुना जिसने illegally park किया?
              
              Generate a Hindi-English pun question for the given sentence: <s> A Hobbitat </s>
              आप उस स्थान को क्या कहते हैं जहाँ एक hobbit रहता है? 
              
              Generate a Hindi-English pun question for the given sentence: <s> {text} </s> 
              <|assistant|>""" 
              for text in texts]

for text in input_text:
    input_ids = tokenizer([text], return_tensors="pt", truncation=True, padding="max_length", max_length=512).to("cuda")
    outputs = model.generate(input_ids = input_ids["input_ids"], attention_mask = input_ids["attention_mask"], max_new_tokens=150, num_beams=5, no_repeat_ngram_size=3, do_sample=False, num_return_sequences=1, temperature=1, top_k=50, top_p=0.95, early_stopping=True)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(text)
    print(output_text.split("<|assistant|>")[1])
    print("----------------------------")
