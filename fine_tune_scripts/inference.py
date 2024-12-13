# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BartForConditionalGeneration, AutoModelForSeq2SeqLM

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("likhithasapu/codemixed-airavata-v2")
model = AutoModelForCausalLM.from_pretrained("likhithasapu/codemixed-airavata-v2", device_map="auto")

# texts = ["My watch is stuck between 2 and 2.30. It's a do or die situation.",
#          "What did the American milk say to the Indian milk? “What’s up दूध?",
#          "A daughter is the perfect child",
#          "I really don't care who takes bath daily",
#          "What does Desi police do to people who steal eggs?"]

texts = [
         "What do you call a person who is outside a door and has no arms nor legs?",
         "Which Star Trek character is a member of the magic circle?", 
         "What's the difference between a bullet and a human?",
         "Why was the Ethiopian baby crying?",
         "What's the difference between a married man and a bachelor?",
         "Why are there so many blood cells in female prisons?",
         "How do you call it when an egg is on point?",
         "Where'd the dog who lost his tail go to get a new one?"
        ]

input_text = [f"""<|user|> 
              Translate the English sentence to Hindi-English sentence: <s> What did 1 say to 7? </s>
              1 ने 7 से क्या कहा?
              
              Translate the English sentence to Hindi-English sentence: <s> Which Star Trek character is a member of the magic circle? </s>
              Which स्टार Trek character magic circle का सदस्य है?
              
              Translate the English sentence to Hindi-English sentence: <s> What's the difference between a married man and a bachelor? </s>
              A married man और a bachelor के बीच क्या difference है।
              
              Translate the English sentence to Hindi-English sentence: <s> What's the opposite of william? </s>
              The opposite of william क्या है?
              
              Translate the English sentence to Hindi-English sentence: <s> Why does the Canadian English alphabet have 52 letters? </s>
              Canadian English alphabet में 52 letters क्यों हैं?
              
              Translate the English sentence to Hindi-English sentence: <s> {text} </s> 
              <|assistant|>""" 
              for text in texts]

for text in input_text:
    input_ids = tokenizer([text], return_tensors="pt", truncation=True, padding="max_length", max_length=512).to("cuda")
    outputs = model.generate(input_ids = input_ids["input_ids"], attention_mask = input_ids["attention_mask"], max_new_tokens=150, num_beams=5, no_repeat_ngram_size=3, do_sample=False, num_return_sequences=1, temperature=0.5, top_k=50, top_p=0.95, early_stopping=True)
    # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(text)
    print(output_text.split("<|assistant|>")[1])
    print("----------------------------")

# ['What do you call a person जो door के बाहर खड़ा है और has no arms या legs ', 
# 'Which स्टार Trek character magic circle का सदस्य है? ', 
# 'a bullet और a human के बीच क्या फर्क है। ', 
# 'Why was the Ethiopian baby रो रही थी। ', 
# 'a married man और a bachelor के बीच क्या difference है।', 
# 'Why are there so many रक्त कोशिकाएं in female कैदियों ', 
# 'How do आप call इसे when an अंडा is on point? ', 
# "Where'd कुत्ते ने lost his tail go to get एक नया one "]
